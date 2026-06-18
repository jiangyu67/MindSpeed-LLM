# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT with optional activation outlier probing."""

import os
import re
import json
import time
from pathlib import Path
from functools import partial
from typing import Union

import torch
from mindspeed_llm import megatron_adaptor
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from mindspeed_llm.training.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from mindspeed_llm.training.utils import set_mtp_batch_list, get_mtp_batch_list
from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank


# =========================
# Activation probing utils
# =========================

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float_list(name: str, default: str):
    raw = os.getenv(name, default).strip()
    if not raw:
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _env_int_set(name: str, default: str = ""):
    raw = os.getenv(name, default).strip()
    if not raw:
        return None
    result = set()
    for x in raw.split(","):
        x = x.strip()
        if x:
            result.add(int(x))
    return result


def _safe_dist_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _safe_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _extract_first_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for x in obj:
            t = _extract_first_tensor(x)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for _, v in obj.items():
            t = _extract_first_tensor(v)
            if t is not None:
                return t
    return None


class ActivationProbe:
    def __init__(self):
        self.enabled = _env_flag("ACTIVATION_PROBE_ENABLE", "0")
        self.output_dir = Path(os.getenv("ACTIVATION_PROBE_DIR", "logs/activation_probe"))
        self.first_n = _env_int("ACTIVATION_PROBE_FIRST_N", 2)
        self.last_n = _env_int("ACTIVATION_PROBE_LAST_N", 2)
        self.sample_size = _env_int("ACTIVATION_PROBE_SAMPLE_SIZE", 200000)
        self.num_bins = _env_int("ACTIVATION_PROBE_BINS", 201)
        self.save_values = _env_flag("ACTIVATION_PROBE_SAVE_VALUES", "0")
        self.thresholds = _env_float_list("ACTIVATION_PROBE_THRESHOLDS", "6,8,10,20,50,100")
        self.target_iters = _env_int_set("ACTIVATION_PROBE_ITERS", "")
        self.include_pattern = os.getenv("ACTIVATION_PROBE_INCLUDE", "").strip()
        self.exclude_pattern = os.getenv("ACTIVATION_PROBE_EXCLUDE", "").strip()

        self.handles = []
        self.current_iter = -1
        self.selected_modules = []
        self._registered = False

        # 不要在分布式初始化前访问这些 rank
        self.rank = 0
        self.world_size = 1
        self.pp_rank = 0
        self.tp_rank = 0
        self.dp_rank = 0

        self.log_path = None
        self.jsonl_path = None

    def refresh_ranks(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

            try:
                self.pp_rank = mpu.get_pipeline_model_parallel_rank()
            except Exception:
                self.pp_rank = 0

            try:
                self.tp_rank = mpu.get_tensor_model_parallel_rank()
            except Exception:
                self.tp_rank = 0

            try:
                self.dp_rank = mpu.get_data_parallel_rank()
            except Exception:
                self.dp_rank = 0

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.output_dir / (
                f"activations_rank{self.rank:03d}_pp{self.pp_rank:02d}_tp{self.tp_rank:02d}_dp{self.dp_rank:02d}.log"
            )
            self.jsonl_path = self.output_dir / (
                f"activations_rank{self.rank:03d}_pp{self.pp_rank:02d}_tp{self.tp_rank:02d}_dp{self.dp_rank:02d}.jsonl"
            )

    def set_iteration(self, iteration: int):
        self.current_iter = int(iteration)

    def should_capture(self) -> bool:
        if not self.enabled:
            return False
        if self.target_iters is None:
            return True
        return self.current_iter in self.target_iters

    def _match_layer_index(self, name: str):
        patterns = [
            r"(?:^|\.)(?:decoder|layers|transformer_layers)\.(\d+)(?:\.|$)",
            r"(?:^|\.)(?:language_model\.encoder\.layers)\.(\d+)(?:\.|$)",
            r"(?:^|\.)(?:module\.decoder\.layers)\.(\d+)(?:\.|$)",
        ]
        for p in patterns:
            m = re.search(p, name)
            if m:
                return int(m.group(1))
        return None

    def _is_embedding_like(self, name: str) -> bool:
        keys = ["embedding", "word_embeddings", "position_embeddings", "tok_embeddings", "input_embedding"]
        return any(k in name for k in keys)

    def _is_output_like(self, name: str) -> bool:
        keys = ["final_layernorm", "final_norm", "output_layer", "lm_head", "output"]
        return any(k in name for k in keys)

    def _local_to_global_layer_index(self, local_layer_idx: int):
        """
        把当前 pipeline stage 内部的本地 layer 编号，映射到全局 layer 编号。
        该实现适用于当前这种均匀切分场景：
          global_layer = pp_rank * layers_per_stage + local_layer
        例如：
          num_layers=28, pp=4
          每个 stage 7 层
          pp=3 的 local 0 -> global 21
        """
        try:
            args = get_args()
            pp_size = getattr(args, "pipeline_model_parallel_size", 1)
            num_layers = getattr(args, "num_layers", None)

            if num_layers is None or pp_size <= 1:
                return local_layer_idx

            if num_layers % pp_size != 0:
                # 非均匀切分时，这个简单公式不一定成立
                # 这里保守返回 None，避免打印错误的全局编号
                return None

            layers_per_stage = num_layers // pp_size
            return self.pp_rank * layers_per_stage + local_layer_idx
        except Exception:
            return None

    def _select_modules(self, model):
        named_modules = list(model.named_modules())
        layer_modules = []
        embedding_modules = []
        output_modules = []

        for name, module in named_modules:
            if not name:
                continue

            if self.include_pattern and re.search(self.include_pattern, name) is None:
                continue
            if self.exclude_pattern and re.search(self.exclude_pattern, name) is not None:
                continue

            layer_idx = self._match_layer_index(name)
            if layer_idx is not None:
                # 尽量抓 block 级，避免抓太多子模块
                if (
                    name.endswith(str(layer_idx))
                    or name.endswith(f"layers.{layer_idx}")
                    or name.endswith(f"decoder.layers.{layer_idx}")
                ):
                    global_layer_idx = self._local_to_global_layer_index(layer_idx)
                    layer_modules.append({
                        "name": name,
                        "module": module,
                        "local_layer_idx": layer_idx,
                        "global_layer_idx": global_layer_idx,
                    })
                continue

            if self._is_embedding_like(name):
                embedding_modules.append({
                    "name": name,
                    "module": module,
                    "local_layer_idx": None,
                    "global_layer_idx": None,
                })
                continue

            if self._is_output_like(name):
                output_modules.append({
                    "name": name,
                    "module": module,
                    "local_layer_idx": None,
                    "global_layer_idx": None,
                })
                continue

        layer_modules = sorted(layer_modules, key=lambda x: (x["local_layer_idx"], x["name"]))

        selected = []
        selected_names = set()

        for item in embedding_modules[:2]:
            if item["name"] not in selected_names:
                selected.append(item)
                selected_names.add(item["name"])

        for item in layer_modules[:self.first_n]:
            if item["name"] not in selected_names:
                selected.append(item)
                selected_names.add(item["name"])

        for item in layer_modules[-self.last_n:]:
            if item["name"] not in selected_names:
                selected.append(item)
                selected_names.add(item["name"])

        for item in output_modules[:4]:
            if item["name"] not in selected_names:
                selected.append(item)
                selected_names.add(item["name"])

        return selected

    def register(self, model):
        if not self.enabled or self._registered:
            return

        self.refresh_ranks()
        self.selected_modules = self._select_modules(model)

        if len(self.selected_modules) == 0:
            self._append_text(f"[rank={self.rank}] No modules selected for activation probe.\n")
            self._registered = True
            return

        selected_names = []
        for item in self.selected_modules:
            name = item["name"]
            if item["local_layer_idx"] is not None:
                name = (
                    f"{name} "
                    f"(local_layer={item['local_layer_idx']}, global_layer={item['global_layer_idx']})"
                )
            selected_names.append(name)

        self._append_text(
            "[ActivationProbe] rank={} pp={} tp={} dp={} thresholds={} selected_modules=\n{}\n".format(
                self.rank,
                self.pp_rank,
                self.tp_rank,
                self.dp_rank,
                self.thresholds,
                "\n".join(f"  - {x}" for x in selected_names),
            )
        )

        for item in self.selected_modules:
            handle = item["module"].register_forward_hook(
                self._make_hook(
                    module_name=item["name"],
                    local_layer_idx=item["local_layer_idx"],
                    global_layer_idx=item["global_layer_idx"],
                )
            )
            self.handles.append(handle)

        self._registered = True

    def _append_text(self, text: str):
        if not self.enabled:
            return
        if self.log_path is None:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text)

    def _make_hook(self, module_name: str, local_layer_idx=None, global_layer_idx=None):
        def hook(module, inputs, output):
            if not self.should_capture():
                return

            try:
                with torch.no_grad():
                    tensor = _extract_first_tensor(output)
                    if tensor is None or not torch.is_floating_point(tensor):
                        return

                    x = tensor.detach().float()
                    numel = x.numel()
                    if numel == 0:
                        return

                    abs_x = x.abs()
                    mean = x.mean().item()
                    std = x.std(unbiased=False).item()
                    min_v = x.min().item()
                    max_v = x.max().item()
                    max_abs = abs_x.max().item()

                    outlier_counts = {}
                    outlier_ratios = {}
                    for th in self.thresholds:
                        cnt = int((abs_x > th).sum().item())
                        ratio = float(cnt) / float(numel)
                        outlier_counts[str(th)] = cnt
                        outlier_ratios[str(th)] = ratio

                    layer_info = ""
                    if local_layer_idx is not None:
                        layer_info += f" local_layer={local_layer_idx}"
                    if global_layer_idx is not None:
                        layer_info += f" global_layer={global_layer_idx}"

                    msg = (
                        f"[ActivationProbe] "
                        f"iter={self.current_iter} "
                        f"rank={self.rank} pp={self.pp_rank} tp={self.tp_rank} dp={self.dp_rank} "
                        f"module={module_name}{layer_info} "
                        f"shape={list(x.shape)} dtype={tensor.dtype} "
                        f"mean={mean:.6e} std={std:.6e} min={min_v:.6e} max={max_v:.6e} max_abs={max_abs:.6e} "
                        f"outlier_counts_abs_gt={outlier_counts} "
                        f"outlier_ratios_abs_gt={outlier_ratios}\n"
                    )
                    self._append_text(msg)

            except Exception as e:
                self._append_text(
                    f"[ActivationProbe][ERROR] iter={self.current_iter} rank={self.rank} module={module_name} error={repr(e)}\n"
                )

        return hook
    
# global singleton
ACTIVATION_PROBE = ActivationProbe()
GLOBAL_FORWARD_ITER = 0


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if not args.use_legacy_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    # register activation hooks once model is built
    ACTIVATION_PROBE.register(model)
    return model


def get_batch(data_iterator):
    """Generate a batch."""

    args = get_args()

    is_middle_stage = not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage())
    pretrain_not_tnd_flags = not args.is_instruction_dataset and not args.reset_attention_mask
    if pretrain_not_tnd_flags and is_middle_stage:
        return (None,) * 5

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    if args.return_document_ids and mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == 0:
        print("current idx: {}, current rank: {}, data_parallel_rank: {}, document_ids: {}".format(batch['idx'], torch.distributed.get_rank(), mpu.get_data_parallel_rank(), batch['document_ids']))
        batch.pop('document_ids', None)
        batch.pop('idx', None)

    # get batch_list for mtp_block
    if args.mtp_num_layers:
        mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
        set_mtp_batch_list(mtp_batch_list)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    try:
        if args.enable_elastic_training:
            from mindspeed_llm.core.high_availability import elastic_training_common
            if not elastic_training_common.zit_scale_in_running_state():
                torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
        else:
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    except Exception:
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    global GLOBAL_FORWARD_ITER

    args = get_args()
    timers = get_timers()

    # Set current iteration for activation probe before model forward
    ACTIVATION_PROBE.set_iteration(GLOBAL_FORWARD_ITER)

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    if args.use_legacy_models:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)

    GLOBAL_FORWARD_ITER += 1
    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def main():
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    if ACTIVATION_PROBE.enabled:
        print_rank_0(
            "[ActivationProbe] enabled. "
            f"dir={ACTIVATION_PROBE.output_dir}, first_n={ACTIVATION_PROBE.first_n}, "
            f"last_n={ACTIVATION_PROBE.last_n}, sample_size={ACTIVATION_PROBE.sample_size}, "
            f"bins={ACTIVATION_PROBE.num_bins}, save_values={ACTIVATION_PROBE.save_values}, "
            f"target_iters={sorted(list(ACTIVATION_PROBE.target_iters)) if ACTIVATION_PROBE.target_iters is not None else 'ALL'}, "
            f"thresholds={ACTIVATION_PROBE.thresholds}"
        )

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step
    )


if __name__ == "__main__":
    main()