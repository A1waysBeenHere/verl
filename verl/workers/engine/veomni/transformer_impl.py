import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from typing import Any, Callable, Optional
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_torch_device,
)
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.dataset.dataset_utils import DatasetPadMode
from ..base import BaseEngine, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from veomni.distributed import parallel_state
from veomni.models.auto import build_tokenizer, build_processor, build_foundation_model
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class VeomniEngine(BaseEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the FSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with FSDP and model settings.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = dist.get_rank()

        parallel_state.init_parallel_state(
            dp_size=self.engine_config.data_parallel_size,
            dp_replicate_size=self.engine_config.data_parallel_replicate_size,
            dp_shard_size=self.engine_config.data_parallel_shard_size,
            tp_size=self.engine_config.tensor_parallel_size,
            ep_size=self.engine_config.expert_parallel_size,
            pp_size=self.engine_config.pipeline_parallel_size,
            cp_size=self.engine_config.context_parallel_size,
            ulysses_size=self.engine_config.ulysses_parallel_size,
            dp_mode=self.engine_config.data_parallel_mode,            
        )

        self.use_remove_padding = self.model_config.use_remove_padding

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        # if self.engine_config.entropy_from_logits_with_chunking:
        #     entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        # else:
        #     entropy_from_logits = verl_F.entropy_from_logits

        # self.compute_entropy_from_logits = (
        #     torch.compile(entropy_from_logits, dynamic=True)
        #     if self.engine_config.use_torch_compile  #  use torch compile by default
        #     else entropy_from_logits
        # )


    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """

        self.checkpoint_manager = build_checkpointer(dist_backend=self.engine_config.data_parallel_mode, ckpt_manager=self.engine_config.ckpt_manager)
        # This is used to import external_lib into the huggingface systems
        self.model = build_foundation_model(
            config_path=self.model_config.config_path,
            weights_path=self.model_config.model_path,
            torch_dtype="float32" if self.engine_config.enable_mixed_precision else "bfloat16",
            attn_implementation=self.model_config.attn_implementation,
            moe_implementation=self.model_config.moe_implementation,
            init_device=self.engine_config.init_device,
            force_use_huggingface=self.model_config.force_use_huggingface,
        )

        model_config = self.model.config

        get_optimizer_pre_hook = getattr(self.model, "get_optimizer_pre_hook", None)
        self.model = build_parallelize_model(
            self.model,
            init_device=self.engine_config.init_device,
            weights_path=self.model_config.model_path,
            enable_full_shard=self.engine_config.enable_full_shard,
            enable_mixed_precision=self.engine_config.enable_mixed_precision,
            enable_gradient_checkpointing=self.engine_config.enable_gradient_checkpointing,
            enable_fsdp_offload=self.engine_config.enable_fsdp_offload,
            basic_modules=self.model._no_split_modules + self.model_config.basic_modules,
            enable_reentrant=self.engine_config.enable_reentrant,
            enable_forward_prefetch=self.engine_config.enable_forward_prefetch,
        )

        self.optimizer = build_optimizer(
            self.model,
            lr=self.engine_config.lr,
            weight_decay=self.engine_config.weight_decay,
            fused=True,
            optimizer_type=self.engine_config.optimizer,
        )
        if get_optimizer_pre_hook is not None:
            optimizer_pre_hook = get_optimizer_pre_hook(self.model, model_config, self.engine_config.data_parallel_mode)
            self.optimizer.register_step_pre_hook(optimizer_pre_hook)

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=self.engine_config.train_steps * self.engine_config.num_train_epochs,
            lr=self.engine_config.lr,
            lr_min=self.engine_config.lr_min,
            lr_decay_style=self.engine_config.lr_decay_style,
            lr_decay_ratio=self.engine_config.lr_decay_ratio,
            lr_warmup_ratio=self.engine_config.lr_warmup_ratio,
            lr_start=self.engine_config.lr_start,
        )

        if self.engine.load_checkpoint_path:
            state = {"model": self.model, "optimizer": self.optimizer, "extra_state": {}}  # cannot be None
            self.checkpoint_manager.load(self.engine.load_checkpoint_path, state)
            global_step = state["extra_state"]["global_step"]
            start_epoch = global_step // self.engine.train_steps
            start_step = global_step % self.engine.train_steps
            self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            # train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
            # environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
            torch.set_rng_state(state["extra_state"]["torch_rng_state"])
            # if start_step == 0:  # resume at the end of epoch
            #     iter(train_dataloader)  # clear resume state and prefetch data

            dist.barrier()
        
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.engine.enable_activation_offload, self.engine.enable_gradient_checkpointing, self.engine.activation_gpu_limit
        )

        self.model.train()
    

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.checkpoint_manager.save(
            local_path, global_step
        )


    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        self.checkpoint_manager.load(
            local_path
        )

    def train_mode(self):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        raise NotImplementedError

    def eval_mode(self):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        """
        Zero the gradients of the optimizer.
        """
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Perform an optimization step using the optimizer.
        """
        if hasattr(self.model, "clip_grad_norm_"):
            _gn = self.model.clip_grad_norm_(self.engine.max_grad_norm)
            grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
        else:
            # logger.info_rank0(
            #     "Can NOT find regitsered clip_grad_norm_ method in the model, using PyTorch default implementation.."
            # )
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.engine.max_grad_norm)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]  # only return the first group
        return lr
    
    # Need Fix
    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]

        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        # args used to get outputs
        output_args = {}

        if use_remove_padding:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
                position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            # if self.use_ulysses_sp:
            #     is_vlm_model = hasattr(getattr(self.module, "module", self.module).config, "vision_config")
            #     if is_vlm_model:
            #         # vlm model's inputs will be sliced after embedding
            #         input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
            #             input_ids_rmpad,
            #             position_ids_rmpad=position_ids_rmpad,
            #             sp_size=self.ulysses_sequence_parallel_size,
            #         )
            #     else:
            #         input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
            #             input_ids_rmpad,
            #             position_ids_rmpad=position_ids_rmpad,
            #             sp_size=self.ulysses_sequence_parallel_size,
            #         )
            #     input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
            #         input_ids_rmpad_rolled,
            #         position_ids_rmpad=None,
            #         sp_size=self.ulysses_sequence_parallel_size,
            #     )

            #     output_args["pad_size"] = pad_size

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled

            # only pass input_ids and position_ids to enable flash_attn_varlen

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                loss_mask = micro_batch["loss_mask"]

                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                )

                attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
                attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
                attention_mask = torch.nested.to_padded_tensor(
                    attention_mask, padding=0, output_size=(batch_size, max_seq_len)
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature
            extra_args["return_dict"] = True

        model_inputs.update(multi_modal_inputs)
        model_inputs.update(extra_args)

        return model_inputs, output_args
    
    # Need Fix
    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

        model_output = {}

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]

            if use_fused_kernels:
                log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
            else:
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                # compute entropy
                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

            # gather log_prob if sp > 1
            # if self.use_ulysses_sp:
            #     pad_size = output_args["pad_size"]

            #     # gather and unpad for the ulysses sp
            #     log_probs = gather_outputs_and_unpad(
            #         log_probs,
            #         gather_dim=0,
            #         unpad_dim=0,
            #         padding_size=pad_size,
            #     )
            #     if calculate_entropy:
            #         entropy_rmpad = gather_outputs_and_unpad(
            #             entropy_rmpad,
            #             gather_dim=0,
            #             unpad_dim=0,
            #             padding_size=pad_size,
            #         )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:  # not using rmpad and no ulysses sp
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:
                logits = output.logits
                logits.div_(temperature)

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                    # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output
    
    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only, mbs_len):
        device_name = get_device_name()
        # actually, we should avoid assigning like this...
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.model(
                **model_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch
            )
            loss = raw_output.loss.mean() / mbs_len
            metrics = {"loss": loss.detach().item()}

            # if loss_function is not None:
            #     loss, metrics = loss_function(
            #         model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
            #     )
            # else:
            #     assert forward_only, "forward_only must be True when loss_function is None"
            #     loss = torch.tensor(1.0, device=device_name)
            #     metrics = {}
            

            output = {
                "model_output": model_output,
                "loss": loss,
                "metrics": metrics,
            }

            return loss, output

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        """
        Perform a forward pass and optionally a backward pass on a batch of data.

        Args:
            data: The input data for the forward pass, typically containing tensors and metadata.
            loss_function: The loss function to optimize. See `verl.workers.roles.utils.losses` for examples.
            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.

        Returns:
            Any: The output of the forward pass, which can be used for loss computation or other purposes.
        """
        tu.assign_non_tensor(data, sp_size=parallel_state._PARALLEL_STATE.ulysses_size)

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        for micro_batch in micro_batches:
            with self.model_fwd_context:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only, mbs_len=len(micro_batches))
            if not forward_only:
                    global_bsz = data["global_batch_size"]
                    local_micro_bsz = micro_batch.batch_size[0]
                    # metrics contain the output, loss is dummy
                    loss_scale_factor = local_micro_bsz / (global_bsz / self.get_data_parallel_size())
                    # scale loss
                    loss = loss * loss_scale_factor
                    loss.backward()

            output_lst.append(meta_info)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    

    def get_per_tensor_param(self):
        raise NotImplementedError

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // parallel_state._PARALLEL_STATE.ulysses_size

    def get_data_parallel_rank(self):
        if parallel_state._PARALLEL_STATE.ulysses_size > 1:
            return parallel_state._PARALLEL_STATE.device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_group(self):
        if parallel_state._PARALLEL_STATE.ulysses_size > 1:
            return parallel_state._PARALLEL_STATE.device_mesh["dp"]
        else:
            return torch.distributed.group.WORLD

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        raise NotImplementedError

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        if parallel_state._PARALLEL_STATE.ulysses_size > 1:
            is_collect = parallel_state._PARALLEL_STATE.device_mesh["ulysses"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect
    

# TODO: 
# Figure out if it is necessary in VeomniEngine, or we can use
#  CPUOffload to achieve auto offload/load operations.
class EngineTrainModeCtx:
    def __init__(self, engine: VeomniEngine):
        self.engine = engine

    def __enter__(self):
        assert isinstance(self.engine, VeomniEngine)
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


# TODO: 
# Figure out if it is necessary in VeomniEngine, or we can use
#  CPUOffload to achieve auto offload/load operations.
class EngineEvalModeCtx:
    def __init__(self, engine: VeomniEngine):
        self.engine = engine

    def __enter__(self):
        assert isinstance(self.engine, VeomniEngine)
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass