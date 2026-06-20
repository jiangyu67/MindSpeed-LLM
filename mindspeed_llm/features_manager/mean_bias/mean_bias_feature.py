"""Averis: mean-bias aware FP4 W4A4G4 quantization feature for MindSpeed-LLM.

Reproduces the method from
"The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training"
(arXiv:2603.10444). The paper names the method *Averis* and applies it to
FP4 (W4A4G4) training: mean-residual splitting is applied to activations
(forward) and output gradients (backward), with the mean vector and the
residual quantized *independently* by a standard blockwise FP4 kernel.

When enabled, this feature patches ``torch.nn.functional.linear`` so that
every 2-D (token x hidden) GeMM goes through :class:`AverisFP4Linear`:
  * Forward  (A4 + W4): activation and weight are both FP4-quantized; the
    activation is first split into a column-mean vector and a zero-mean
    residual, which are quantized independently.
  * Backward (G4): the output gradient is likewise mean-residual split and
    FP4-quantized; weight gradients are FP4-quantized too.
"""
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager

from mindspeed_llm.features_manager.mean_bias.quant_impl import mean_bias_linear


# Module-level state set during register_patches; read by the patched linear.
_STATE = {
    'enabled': False,
    'block_size': 16,
    'min_numel': 1024,
    'orig_linear': F.linear,
    'log_freq': 100,
    'call_count': 0,
    'log_path': None,
}


def _patched_linear(input, weight, bias=None):
    """Replacement for torch.nn.functional.linear with mean-residual FP4 split."""
    if not _STATE['enabled']:
        return _STATE['orig_linear'](input, weight, bias)

    out = mean_bias_linear(
        input, weight, bias=bias,
        block_size=_STATE['block_size'],
        min_numel=_STATE['min_numel'],
        linear_fn=_STATE['orig_linear'],
    )

    # Lightweight logging: periodically record how often the split path fires
    # and the dynamic-range reduction achieved by mean subtraction.
    _STATE['call_count'] += 1
    log_freq = _STATE['log_freq']
    if log_freq > 0 and (_STATE['call_count'] % log_freq == 0) and _STATE['log_path']:
        try:
            if input.ndim == 2 and input.numel() >= _STATE['min_numel']:
                with torch.no_grad():
                    mu = input.mean(dim=0)
                    R = input - mu.unsqueeze(0)
                    orig_max = input.abs().amax().item()
                    resid_max = R.abs().amax().item()
                    mu_norm = mu.norm().item()
                with open(_STATE['log_path'], 'a') as fh:
                    fh.write(
                        f"calls={_STATE['call_count']} shape={tuple(input.shape)} "
                        f"|X|_inf={orig_max:.4f} |R|_inf={resid_max:.4f} "
                        f"ratio={resid_max / max(orig_max, 1e-8):.4f} "
                        f"||mu||={mu_norm:.4f}\n"
                    )
        except Exception:
            pass

    return out


class MeanBiasFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('mean_bias', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--mean-bias', action='store_true', default=False,
                           help='Enable mean-bias aware FP4 activation quantization '
                                '(mean-residual splitting, arXiv:2603.10444)')
        group.add_argument('--mean-bias-block-size', type=int, default=16,
                           help='Block size for blockwise FP4 quantization of the residual')
        group.add_argument('--mean-bias-min-numel', type=int, default=1024,
                           help='Skip mean-residual split for activations smaller than this')
        group.add_argument('--mean-bias-log-freq', type=int, default=100,
                           help='Frequency (linear calls) for logging dynamic-range stats; '
                                '0 disables logging')
        group.add_argument('--mean-bias-log-path', type=str,
                           default='/home/zs/mean_bias/mean_bias.log',
                           help='Path to write mean-bias dynamic-range log')

    def register_patches(self, patch_manager: MindSpeedPatchesManager, args):
        if not getattr(args, 'mean_bias', False):
            return

        # Configure module-level state used by _patched_linear.
        _STATE['enabled'] = True
        _STATE['block_size'] = getattr(args, 'mean_bias_block_size', 16)
        _STATE['min_numel'] = getattr(args, 'mean_bias_min_numel', 1024)
        _STATE['log_freq'] = getattr(args, 'mean_bias_log_freq', 100)
        log_path = getattr(args, 'mean_bias_log_path',
                           '/home/zs/mean_bias/mean_bias.log')
        _STATE['log_path'] = log_path
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except Exception:
            pass

        # Save the true original linear before installing the patch.
        _STATE['orig_linear'] = F.linear

        # Install the global patch on torch.nn.functional.linear. We replace
        # the attribute directly because the patch target is a torch builtin
        # function, not a class method. Most Megatron/PyTorch GeMMs go through
        # ``F.linear`` / ``torch.nn.functional.linear``, so this catches the
        # forward activations of QKV proj, output proj, and FFN up/down layers.
        torch.nn.functional.linear = _patched_linear
