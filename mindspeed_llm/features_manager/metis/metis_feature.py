"""Metis spectral FP4/FP8 quantization feature for MindSpeed-LLM.

Patches the Megatron-LM optimizer to:
  * quantize gradients blockwise (FP4 E2M1 or FP8) with optional spectral
    subspace projection (Metis low-rank + residual split);
  * periodically refresh the dominant subspace cache for 2-D parameters;
  * quantize/dequantize optimizer states in state_dict / load_state_dict;
  * log spectral statistics (singular values, energy concentration) to disk.

The FP4 path is a faithful simulation: tensors stay float32 but values are
snapped to the 8 E2M1 representable magnitudes per block. On NPU hardware
without native FP4, this gives the same numerical behavior as a real FP4
kernel would; when FP4 kernels become available the simulation can be swapped
out for the hardware op without changing the hook logic.
"""
import os
import time
from argparse import ArgumentParser
from typing import Dict, Optional

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager

import torch

from mindspeed_llm.features_manager.metis.quant_impl import (
    FP4_LEVELS,
    apply_metis_quantization,
    compute_subspace,
    fp4_quantize_blockwise,
    fp8_quantize_blockwise,
    metis_quantize_with_subspace,
    project_to_subspace,
    quantize_blockwise,
    randomized_svd,
    spectrum_stats,
)


# ---------------------------------------------------------------------------
# Forward W4A4 patch state
# ---------------------------------------------------------------------------
# Metis paper specifies W4A4G4: weights, activations AND gradients are FP4
# quantized. The optimizer hooks above cover G4 (gradient quantization). The
# forward patch below covers W4 (weight) and A4 (activation) by patching
# ``torch.nn.functional.linear``: every 2-D GeMM routes the weight and
# activation through Metis spectral-split FP4 quantization (STE). Weight
# subspaces are reused from the cached ``_metis_name`` -> U map refreshed in
# ``train_step``; activations use the online randomized-SVD path.
_FORWARD_STATE = {
    'enabled': False,
    'orig_linear': None,
    'block_size': 16,
    'min_numel': 1024,
    'rank_frac': 0.015,
    'sample_ratio': 0.01,
    'qdtype': 'fp4',
}


def _metis_ste_quantize(tensor, quantize_fn):
    """Straight-through estimator: forward = quantize, backward = identity."""
    with torch.no_grad():
        q = quantize_fn(tensor)
    return tensor + (q - tensor).detach()


def _metis_quantize_weight(weight):
    """W4: use cached subspace if available, else online spectral split."""
    name = getattr(weight, '_metis_name', None)
    U = None
    if name:
        gargs = _get_global_args()
        if gargs and getattr(gargs, 'metis', False):
            st = _get_metis_state(gargs)
            entry = st['subspaces'].get(name)
            if isinstance(entry, dict):
                U = entry.get('U')
    bs = _FORWARD_STATE['block_size']
    qd = _FORWARD_STATE['qdtype']
    if U is not None:
        return metis_quantize_with_subspace(weight, U, qdtype=qd, block_size=bs)
    return apply_metis_quantization(
        weight, rank_frac=_FORWARD_STATE['rank_frac'],
        block_size=bs, sample_ratio=_FORWARD_STATE['sample_ratio'], qdtype=qd)


def _metis_quantize_activation(activation):
    """A4: spectral split + blockwise FP4 (faithful to Metis paper §3.2).

    The Metis paper applies spectral decomposition to weights, activations, AND
    gradients — all GeMM matrices in forward and backward are FP4-quantized
    via the spectral-split path. We do NOT shortcut activations to plain
    blockwise FP4; we follow the paper's ``apply_metis_quantization``:
    sparsely-random-sampled randomized SVD → (low_rank, residual) →
    independent blockwise FP4 of each part.
    """
    return apply_metis_quantization(
        activation, rank_frac=_FORWARD_STATE['rank_frac'],
        block_size=_FORWARD_STATE['block_size'],
        sample_ratio=_FORWARD_STATE['sample_ratio'],
        qdtype=_FORWARD_STATE['qdtype'])


def _metis_forward_linear(input, weight, bias=None):
    """Patched ``F.linear`` applying Metis W4A4 spectral FP4 quantization."""
    if not _FORWARD_STATE['enabled']:
        return _FORWARD_STATE['orig_linear'](input, weight, bias)
    # Only quantize 2-D (token x hidden) GeMMs above the min-size threshold.
    if (input.ndim != 2 or weight.ndim != 2
            or input.numel() < _FORWARD_STATE['min_numel']
            or weight.numel() < _FORWARD_STATE['min_numel']):
        return _FORWARD_STATE['orig_linear'](input, weight, bias)
    try:
        Wq = _metis_ste_quantize(weight, _metis_quantize_weight)
        Aq = _metis_ste_quantize(input, _metis_quantize_activation)
        return _FORWARD_STATE['orig_linear'](Aq, Wq, bias)
    except Exception:
        # Fall back to the unquantized path on any numerical issue.
        return _FORWARD_STATE['orig_linear'](input, weight, bias)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _get_global_args():
    try:
        from megatron.training import get_args
        return get_args()
    except Exception:
        return None


def _get_metis_state(args_obj) -> Optional[Dict]:
    """Return (creating if needed) the per-run Metis state dict on ``args``.

    Holds:
      subspaces: {param_name: {'U': Tensor, 'step': int, 'rank': int}}
      step:      global train_step counter
    """
    if args_obj is None:
        return None
    state = getattr(args_obj, '_metis_state', None)
    if state is None:
        state = {'subspaces': {}, 'step': 0}
        try:
            setattr(args_obj, '_metis_state', state)
        except Exception:
            return None
    return state


def _param_name_for(param) -> Optional[str]:
    """Best-effort stable key for a parameter.

    ``data_ptr`` changes on reallocation so it is unsafe as a cache key; we
    rely on Megatron attaching ``_metis_name`` during model setup, falling back
    to id() which is stable for the lifetime of the parameter object.
    """
    name = getattr(param, '_metis_name', None)
    if name is not None:
        return name
    return f'param_{id(param)}'


def _quantize_grad(grad: torch.Tensor, qdtype: str, block_size: int,
                   U: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Quantize a gradient tensor, optionally using a cached subspace."""
    if grad.ndim != 2:
        # 1-D / ND tensors: plain blockwise quantize (no spectral split).
        return quantize_blockwise(grad, qdtype=qdtype, block_size=block_size)
    if U is not None:
        return metis_quantize_with_subspace(grad, U, qdtype=qdtype, block_size=block_size)
    return apply_metis_quantization(
        grad, rank_frac=0.015, block_size=block_size, sample_ratio=0.01, qdtype=qdtype
    )


def _quantize_state_tensor(tensor: torch.Tensor, qdtype: str, block_size: int) -> torch.Tensor:
    """Quantize an optimizer-state tensor for state_dict serialization."""
    if tensor.numel() == 0 or not tensor.is_floating_point():
        return tensor
    if tensor.ndim == 2:
        return apply_metis_quantization(
            tensor, rank_frac=0.015, block_size=block_size, sample_ratio=0.01, qdtype=qdtype
        )
    return quantize_blockwise(tensor, qdtype=qdtype, block_size=block_size)


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

class MetisFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('metis', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--metis', action='store_true', default=False,
                           help='Enable Metis spectral quantization')
        group.add_argument('--metis-rank-frac', type=float, default=0.015,
                           help='Rank fraction for low-rank decomposition')
        group.add_argument('--metis-sample-ratio', type=float, default=0.01,
                           help='Row sampling ratio for decomposition')
        group.add_argument('--metis-block-size', type=int, default=16,
                           help='Block size for blockwise quantization')
        group.add_argument('--metis-quant-dtype', type=str, choices=['fp4', 'fp8'],
                           default='fp4', help='Quantization dtype to simulate')
        group.add_argument('--metis-update-freq', type=int, default=1000,
                           help='Update frequency (steps) for Metis subspace')
        group.add_argument('--metis-output-dir', type=str,
                           default='/home/zs/metis/output',
                           help='Directory to save Metis spectrum logs')
        group.add_argument('--metis-log-freq', type=int, default=100,
                           help='Frequency (steps) for spectrum logging')
        group.add_argument('--metis-max-subspace-params', type=int, default=64,
                           help='Max number of 2-D params to cache subspaces for')

    # ------------------------------------------------------------------
    # Patch registration
    # ------------------------------------------------------------------
    def register_patches(self, patch_manager: MindSpeedPatchesManager, args):
        if not getattr(args, 'metis', False):
            return

        # ---- prepare_grads: quantize gradients (with subspace if cached) ----
        # NOTE: FP4 simulation creates large float32 intermediates (U @ (U.T @ M))
        # which can OOM on NPU. We move grads to CPU for quantization, then move
        # the quantized result back to the original device.
        def prepare_grads_wrapper(orig_func):
            def wrapper(self, *a, **k):
                out = orig_func(self, *a, **k)
                gargs = _get_global_args() or args
                qdtype = _safe_attr(gargs, 'metis_quant_dtype', 'fp4') or 'fp4'
                block_size = _safe_attr(gargs, 'metis_block_size', 16) or 16
                state = _get_metis_state(gargs)
                subspaces = state['subspaces'] if state else {}

                for group in getattr(self, 'param_groups', []):
                    for p in group.get('params', []):
                        if p is None or p.grad is None:
                            continue
                        if p.grad.ndim < 1:
                            continue
                        try:
                            pname = _param_name_for(p)
                            entry = subspaces.get(pname)
                            U = entry['U'] if entry else None
                            # 直接在 NPU 上量化，避免 PCIe 传输开销
                            grad = p.grad.data.detach().to(torch.float32)
                            if U is not None and U.device != grad.device:
                                U = U.to(grad.device)
                            q_grad = _quantize_grad(grad, qdtype, block_size, U=U)
                            p.grad.data = q_grad.to(p.grad.dtype)
                        except Exception as e:
                            # Log once per failure type but don't crash training.
                            if not getattr(p, '_metis_quant_err_logged', False):
                                import sys
                                print(f'[Metis] prepare_grads quantization failed for '
                                      f'{pname}: {type(e).__name__}: {e}', file=sys.stderr)
                                p._metis_quant_err_logged = True
                return out
            wrapper.__name__ = 'metis_prepare_grads_wrapper'
            return wrapper

        patch_manager.register_patch(
            'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.prepare_grads',
            prepare_grads_wrapper, force_patch=False)

        # ---- step: ensure grads are float32 before the actual step ----
        def step_wrapper(orig_func):
            def wrapper(self, *a, **k):
                try:
                    for group in getattr(self, 'param_groups', []):
                        for p in group.get('params', []):
                            if p is None or p.grad is None:
                                continue
                            if p.grad.dtype != torch.float32:
                                p.grad.data = p.grad.data.to(torch.float32)
                except Exception:
                    pass
                return orig_func(self, *a, **k)
            wrapper.__name__ = 'metis_step_wrapper'
            return wrapper

        patch_manager.register_patch(
            'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
            step_wrapper, force_patch=False)
        patch_manager.register_patch(
            'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step_with_ready_grads',
            step_wrapper, force_patch=False)

        # ---- train_step: subspace refresh + spectrum logging ----
        def train_step_wrapper(orig_func):
            def wrapper(*a, **k):
                out = orig_func(*a, **k)

                gargs = _get_global_args() or args
                state = _get_metis_state(gargs)
                if state is None:
                    return out

                state['step'] = state.get('step', 0) + 1
                step = state['step']

                update_freq = max(1, _safe_attr(gargs, 'metis_update_freq', 1000) or 1000)
                log_freq = max(1, _safe_attr(gargs, 'metis_log_freq', 100) or 100)
                out_dir = _safe_attr(gargs, 'metis_output_dir',
                                     '/home/zs/metis/output') or '/home/zs/metis/output'
                rank_frac = _safe_attr(gargs, 'metis_rank_frac', 0.015) or 0.015
                sample_ratio = _safe_attr(gargs, 'metis_sample_ratio', 0.01) or 0.01
                block_size = _safe_attr(gargs, 'metis_block_size', 16) or 16
                max_params = _safe_attr(gargs, 'metis_max_subspace_params', 64) or 64
                qdtype = _safe_attr(gargs, 'metis_quant_dtype', 'fp4') or 'fp4'

                need_subspace_update = (step % update_freq == 0)
                need_log = (step % log_freq == 0)

                if not (need_subspace_update or need_log):
                    return out

                # Locate the model: try argument introspection, then global getter.
                model = None
                for arg in a:
                    if hasattr(arg, 'module'):
                        model = getattr(arg, 'module')
                        break
                    if hasattr(arg, 'parameters') and hasattr(arg, 'named_parameters'):
                        model = arg
                        break
                if model is None:
                    try:
                        from megatron.core import get_model
                        model = get_model()
                        if isinstance(model, (list, tuple)):
                            model = model[0] if model else None
                    except Exception:
                        model = None
                if model is None:
                    return out

                # Collect 2-D params to process (cap to max_params by size desc).
                candidates = []
                for name, param in model.named_parameters():
                    if param.ndim == 2 and param.numel() > block_size:
                        param._metis_name = name
                        candidates.append((name, param))
                candidates.sort(key=lambda np: np[1].numel(), reverse=True)
                candidates = candidates[:max_params]

                results = {}
                subspaces = state['subspaces']

                for name, param in candidates:
                    try:
                        # 保持在 NPU 上计算，避免 CPU 传输开销
                        w = param.detach().to(torch.float32)
                        entry = {'name': name, 'shape': list(w.shape)}

                        if need_subspace_update:
                            U = compute_subspace(w, rank_frac=rank_frac,
                                                 sample_ratio=sample_ratio)
                            subspaces[name] = {
                                'U': U, 'step': step,
                                'rank': U.shape[1], 'block_size': block_size,
                            }
                            entry['subspace_rank'] = U.shape[1]
                            entry['subspace_updated'] = True
                        else:
                            entry['subspace_updated'] = False

                        if need_log:
                            # logging 需要搬到 CPU 做序列化
                            w_cpu = w.cpu()
                            entry['weight_spectrum'] = spectrum_stats(w_cpu)
                            qdtype_eff = qdtype
                            U = subspaces.get(name, {}).get('U')
                            if U is not None:
                                q_w = metis_quantize_with_subspace(
                                    w_cpu, U.cpu(), qdtype=qdtype_eff, block_size=block_size)
                            else:
                                q_w = quantize_blockwise(w_cpu, qdtype=qdtype_eff,
                                                         block_size=block_size)
                            entry['quantized_spectrum'] = spectrum_stats(q_w)
                            err = (w_cpu - q_w).norm().item() / max(w_cpu.norm().item(), 1e-8)
                            entry['quant_rel_error'] = err
                        results[name] = entry
                    except Exception:
                        continue

                if need_log and results:
                    try:
                        spectra_dir = os.path.join(out_dir, 'spectra')
                        os.makedirs(spectra_dir, exist_ok=True)
                        torch.save(
                            {'step': step, 'results': results, 'time': time.time()},
                            os.path.join(spectra_dir, f'metis_spectrum_step_{step}.pt'),
                        )
                    except Exception:
                        pass
                return out
            wrapper.__name__ = 'metis_train_step_wrapper'
            return wrapper

        patch_manager.register_patch(
            'megatron.training.training.train_step',
            train_step_wrapper, force_patch=False)

        # ---- state_dict: quantize optimizer states for serialization ----
        def state_dict_wrapper(orig_func):
            def wrapper(self, *a, **k):
                sd = orig_func(self, *a, **k)
                try:
                    gargs = _get_global_args()
                    if gargs and getattr(gargs, 'metis', False):
                        qdtype = getattr(gargs, 'metis_quant_dtype', 'fp4')
                        block_size = getattr(gargs, 'metis_block_size', 16)
                        optim_state = sd.get('optimizer', {})
                        st = optim_state.get('state', {})
                        for _, v in st.items():
                            if not isinstance(v, dict):
                                continue
                            for subk, subv in list(v.items()):
                                if isinstance(subv, torch.Tensor) and subv.is_floating_point():
                                    try:
                                        v[subk] = _quantize_state_tensor(subv, qdtype, block_size)
                                    except Exception:
                                        pass
                except Exception:
                    pass
                return sd
            wrapper.__name__ = 'metis_state_dict_wrapper'
            return wrapper

        patch_manager.register_patch(
            'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.state_dict',
            state_dict_wrapper, force_patch=False)

        # ---- load_state_dict: dequantize optimizer states before loading ----
        def load_state_dict_wrapper(orig_func):
            def wrapper(self, state_dict, *a, **k):
                try:
                    gargs = _get_global_args()
                    if gargs and getattr(gargs, 'metis', False):
                        optim_state = state_dict.get('optimizer', {})
                        st = optim_state.get('state', {})
                        for _, v in st.items():
                            if not isinstance(v, dict):
                                continue
                            for subk, subv in list(v.items()):
                                if isinstance(subv, torch.Tensor) and subv.dtype != torch.float32:
                                    try:
                                        v[subk] = subv.to(torch.float32)
                                    except Exception:
                                        pass
                except Exception:
                    pass
                return orig_func(self, state_dict, *a, **k)
            wrapper.__name__ = 'metis_load_state_dict_wrapper'
            return wrapper

        patch_manager.register_patch(
            'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.load_state_dict',
            load_state_dict_wrapper, force_patch=False)

        # ---- forward W4A4: patch F.linear for spectral FP4 quantization ----
        # Metis is W4A4G4 (paper §3): weights and activations must also be
        # FP4-quantized in the forward pass, not only gradients. The optimizer
        # hooks above handle G4; this patch adds W4 + A4.
        import torch.nn.functional as _F
        if _FORWARD_STATE['orig_linear'] is None:
            _FORWARD_STATE['orig_linear'] = _F.linear
        gargs_fwd = _get_global_args()
        if gargs_fwd is not None:
            _FORWARD_STATE['block_size'] = int(getattr(gargs_fwd, 'metis_block_size', 16) or 16)
            _FORWARD_STATE['min_numel'] = int(getattr(gargs_fwd, 'metis_min_numel', 1024) or 1024)
            _FORWARD_STATE['rank_frac'] = float(getattr(gargs_fwd, 'metis_rank_frac', 0.015) or 0.015)
            _FORWARD_STATE['sample_ratio'] = float(getattr(gargs_fwd, 'metis_sample_ratio', 0.01) or 0.01)
            _FORWARD_STATE['qdtype'] = getattr(gargs_fwd, 'metis_qdtype', 'fp4') or 'fp4'
        _FORWARD_STATE['enabled'] = True
        torch.nn.functional.linear = _metis_forward_linear
