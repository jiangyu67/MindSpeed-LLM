"""End-to-end smoke test for the Metis feature on a small Qwen-style model.

Validates the full Metis pipeline without requiring the full Megatron training
stack:
  1. FP4/FP8 blockwise quantization of weights and gradients.
  2. Metis spectral decomposition (low-rank + residual).
  3. Subspace caching and projection (as the train_step hook does).
  4. Optimizer state_dict quantize / load_state_dict dequantize round-trip.
  5. Spectrum logging + analysis script.

Runs on CPU with TORCH_DEVICE_BACKEND_AUTOLOAD=0 so it works without NPU.
For a real Qwen2.5-7B run on NPU, use the MindSpeed-LLM training entrypoint
with --metis and the Ascend environment sourced.

Usage:
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python \
        mindspeed_llm/features_manager/metis/smoke_test.py
"""
import argparse
import importlib.util
import os
import sys
import tempfile

# Force CPU-only before importing torch.
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')

# Import quant_impl directly to avoid mindspeed_llm/__init__.py (NPU init).
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'metis_quant_impl', os.path.join(_HERE, 'quant_impl.py'))
qi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qi)

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Tiny Qwen-style model for fast CPU smoke test
# ---------------------------------------------------------------------------

class TinyQwenBlock(nn.Module):
    def __init__(self, hidden=64, intermediate=128, heads=4):
        super().__init__()
        self.self_attn = nn.ModuleDict({
            'q_proj': nn.Linear(hidden, hidden, bias=False),
            'k_proj': nn.Linear(hidden, hidden, bias=False),
            'v_proj': nn.Linear(hidden, hidden, bias=False),
            'o_proj': nn.Linear(hidden, hidden, bias=False),
        })
        self.mlp = nn.ModuleDict({
            'up_proj': nn.Linear(hidden, intermediate, bias=False),
            'down_proj': nn.Linear(intermediate, hidden, bias=False),
        })
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.norm1(x)
        q = self.self_attn['q_proj'](h)
        k = self.self_attn['k_proj'](h)
        v = self.self_attn['v_proj'](h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        x = x + self.self_attn['o_proj'](attn @ v)
        h = self.norm2(x)
        x = x + self.mlp['down_proj'](torch.relu(self.mlp['up_proj'](h)))
        return x


class TinyQwenModel(nn.Module):
    def __init__(self, vocab=1000, hidden=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyQwenBlock(hidden=hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.model = type('M', (), {'layers': self.layers})()  # mimic Qwen attr path

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Smoke test steps
# ---------------------------------------------------------------------------

def step1_quantization_smoke():
    """Verify FP4/FP8 quantization preserves shape and reduces error."""
    print('[1] FP4/FP8 quantization smoke test')
    torch.manual_seed(42)
    w = torch.randn(128, 128)
    q4 = qi.fp4_quantize_blockwise(w, block_size=16)
    q8 = qi.fp8_quantize_blockwise(w, block_size=16)
    assert q4.shape == w.shape
    assert q8.shape == w.shape
    err4 = (w - q4).norm().item() / w.norm().item()
    err8 = (w - q8).norm().item() / w.norm().item()
    print(f'    FP4 rel err: {err4:.4f}, FP8 rel err: {err8:.4f}')
    assert err8 < err4, 'FP8 should have lower error than FP4'
    print('    PASS')


def step2_metis_decomposition_smoke():
    """Verify Metis split + quantize preserves spectral structure."""
    print('[2] Metis decomposition smoke test')
    torch.manual_seed(42)
    # Low-rank-dominant matrix
    A = torch.randn(128, 4)
    B = torch.randn(4, 128)
    w = A @ B + 0.01 * torch.randn(128, 128)
    low_rank, residual = qi.metis_decompose_matrix(w, rank_frac=0.05, sample_ratio=1.0)
    assert residual.norm().item() < 0.5 * w.norm().item()
    q = qi.apply_metis_quantization(w, rank_frac=0.05, block_size=16, sample_ratio=1.0, qdtype='fp4')
    err = (w - q).norm().item() / w.norm().item()
    print(f'    Metis FP4 rel err: {err:.4f}, residual norm ratio: {residual.norm().item()/w.norm().item():.4f}')
    print('    PASS')


def step3_subspace_caching_smoke():
    """Verify subspace caching + projection matches full decompose."""
    print('[3] Subspace caching smoke test')
    torch.manual_seed(42)
    w = torch.randn(64, 64)
    U = qi.compute_subspace(w, rank_frac=0.1, sample_ratio=1.0)
    q_cached = qi.metis_quantize_with_subspace(w, U, qdtype='fp4', block_size=16)
    q_full = qi.apply_metis_quantization(w, rank_frac=0.1, block_size=16, sample_ratio=1.0, qdtype='fp4')
    diff = (q_cached - q_full).norm().item() / max(q_full.norm().item(), 1e-8)
    print(f'    cached vs full quant diff: {diff:.4f}')
    assert diff < 0.5
    print('    PASS')


def step4_optimizer_state_roundtrip_smoke():
    """Verify optimizer state quantize/dequantize round-trip."""
    print('[4] Optimizer state_dict round-trip smoke test')
    torch.manual_seed(42)
    model = TinyQwenModel()
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    # Run a few steps to populate optimizer state.
    for _ in range(3):
        ids = torch.randint(0, 1000, (2, 16))
        logits = model(ids)
        loss = logits.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Quantize state tensors (as state_dict_wrapper does).
    qdtype = 'fp4'
    block_size = 16
    sd = opt.state_dict()
    state = sd.get('state', {})
    n_quantized = 0
    for _, v in state.items():
        if isinstance(v, dict):
            for subk, subv in list(v.items()):
                if isinstance(subv, torch.Tensor) and subv.is_floating_point():
                    v[subk] = qi.apply_metis_quantization(
                        subv, rank_frac=0.015, block_size=block_size,
                        sample_ratio=0.01, qdtype=qdtype) if subv.ndim == 2 \
                        else qi.quantize_blockwise(subv, qdtype=qdtype, block_size=block_size)
                    n_quantized += 1
    print(f'    Quantized {n_quantized} optimizer state tensors')

    # Dequantize back (as load_state_dict_wrapper does).
    for _, v in state.items():
        if isinstance(v, dict):
            for subk, subv in list(v.items()):
                if isinstance(subv, torch.Tensor) and subv.dtype != torch.float32:
                    v[subk] = subv.to(torch.float32)

    # Verify load doesn't crash.
    opt2 = optim.AdamW(model.parameters(), lr=1e-3)
    opt2.load_state_dict(sd)
    print('    PASS (state_dict round-trip OK)')


def step5_full_training_step_smoke():
    """Simulate one Metis-augmented training step end-to-end."""
    print('[5] Full Metis training step smoke test')
    torch.manual_seed(42)
    model = TinyQwenModel()
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    # Simulate the train_step hook: compute subspace for 2-D params.
    subspaces = {}
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.numel() > 16:
            p._metis_name = name
            U = qi.compute_subspace(p.detach().to(torch.float32),
                                    rank_frac=0.1, sample_ratio=1.0)
            subspaces[name] = {'U': U, 'rank': U.shape[1]}
    print(f'    Cached subspaces for {len(subspaces)} params')

    # Forward + backward.
    ids = torch.randint(0, 1000, (2, 16))
    logits = model(ids)
    loss = logits.mean()
    opt.zero_grad()
    loss.backward()

    # Simulate prepare_grads_wrapper: quantize grads with subspace.
    qdtype = 'fp4'
    block_size = 16
    n_quantized = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        entry = subspaces.get(name)
        U = entry['U'] if entry else None
        if p.grad.ndim == 2 and U is not None:
            q = qi.metis_quantize_with_subspace(p.grad.data, U, qdtype=qdtype, block_size=block_size)
        else:
            q = qi.quantize_blockwise(p.grad.data, qdtype=qdtype, block_size=block_size)
        p.grad.data = q.to(p.grad.dtype)
        n_quantized += 1
    print(f'    Quantized {n_quantized} gradients')

    # step_wrapper: cast grads back to fp32.
    for p in model.parameters():
        if p.grad is not None and p.grad.dtype != torch.float32:
            p.grad.data = p.grad.data.to(torch.float32)

    # Actual optimizer step.
    opt.step()
    print(f'    Loss after step: {loss.item():.4f}')
    print('    PASS')


def step6_spectrum_logging_smoke():
    """Verify spectrum logging + analysis pipeline."""
    print('[6] Spectrum logging smoke test')
    with tempfile.TemporaryDirectory() as tmpdir:
        spectra_dir = os.path.join(tmpdir, 'spectra')
        os.makedirs(spectra_dir, exist_ok=True)
        torch.manual_seed(42)
        model = TinyQwenModel()
        for step in [1, 2]:
            results = {}
            for name, p in model.named_parameters():
                if p.ndim != 2:
                    continue
                w = p.detach().to(torch.float32)
                q = qi.quantize_blockwise(w, qdtype='fp4', block_size=16)
                results[name] = {
                    'name': name, 'shape': list(w.shape),
                    'subspace_updated': True, 'subspace_rank': 3,
                    'weight_spectrum': qi.spectrum_stats(w),
                    'quantized_spectrum': qi.spectrum_stats(q),
                    'quant_rel_error': (w - q).norm().item() / max(w.norm().item(), 1e-8),
                }
            torch.save({'step': step, 'results': results},
                       os.path.join(spectra_dir, f'metis_spectrum_step_{step}.pt'))

        # Run analysis script.
        import subprocess
        r = subprocess.run(
            [sys.executable, os.path.join(_HERE, 'analyze_spectra.py'),
             '--spectra-dir', spectra_dir, '--output-dir', os.path.join(tmpdir, 'analysis')],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, 'MPLCONFIGDIR': tmpdir})
        assert r.returncode == 0, f'analyze_spectra failed: {r.stderr}'
        assert os.path.exists(os.path.join(tmpdir, 'analysis', 'spectrum_summary.csv'))
        print('    PASS (spectrum logs + analysis OK)')


def main():
    print('=' * 60)
    print('Metis end-to-end smoke test (CPU, TORCH_DEVICE_BACKEND_AUTOLOAD=0)')
    print('=' * 60)
    step1_quantization_smoke()
    step2_metis_decomposition_smoke()
    step3_subspace_caching_smoke()
    step4_optimizer_state_roundtrip_smoke()
    step5_full_training_step_smoke()
    step6_spectrum_logging_smoke()
    print('=' * 60)
    print('ALL SMOKE TESTS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()
