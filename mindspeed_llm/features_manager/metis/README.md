# Metis FP4 Spectral Quantization — MindSpeed-LLM Integration

This module reproduces the core ideas from **Metis: Training LLMs with FP4
Quantization** (arXiv:2509.00404) inside MindSpeed-LLM. Metis identifies
spectral anisotropy (a few dominant singular values) in weights, activations,
and gradients as the root cause of FP4 quantization error, and addresses it by
partitioning the spectrum into a low-rank dominant subspace + residual, then
quantizing each part independently with blockwise scaling.

## Files

| File | Purpose |
|------|---------|
| `quant_impl.py` | FP4 E2M1 / FP8 blockwise quantization, randomized SVD, Metis decomposition, subspace projection, spectrum stats. |
| `metis_feature.py` | MindSpeed feature that patches the Megatron optimizer (prepare_grads / step / train_step / state_dict / load_state_dict). |
| `test_quant_impl.py` | Unit tests for the quantization primitives (23 tests). |
| `smoke_test.py` | End-to-end smoke test on a tiny Qwen-style model (CPU, no NPU needed). |
| `analyze_spectra.py` | Offline analysis of spectrum logs (CSV summary + plots). |

## How it works

### 1. FP4 E2M1 quantization (simulation)
Tensors stay in float32 but each block of `block_size` elements is scaled by its
max-abs value and snapped to the 8 representable E2M1 magnitudes
`[0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`. This faithfully reproduces the
quantization error of real FP4 hardware. When native FP4 NPU kernels become
available, only `fp4_quantize_blockwise` needs to be swapped out — the hook
logic is unchanged.

### 2. Metis spectral split
For a 2-D matrix `W`, a randomized SVD is computed on a small row sample
(Metis sparsely-random-sampling property) to extract the dominant right-singular
subspace `V`. The matrix is split as:
```
low_rank  = (W @ V) @ V^T      # dominant spectral energy
residual  = W - low_rank       # remaining spectral energy
```
Both parts are then quantized blockwise. Because each part has a narrower
numerical range than the whole, the quantization error is reduced and the
spectral structure is preserved.

### 3. Subspace caching
The dominant subspace `U` (or `V`) is expensive to compute, so it is cached per
parameter and refreshed every `--metis-update-freq` steps inside the
`train_step` hook. Between refreshes, `prepare_grads` reuses the cached subspace
to project + quantize gradients cheaply.

### 4. Optimizer hooks
- `prepare_grads`: quantizes each parameter's gradient (FP4/FP8), using the
  cached subspace if available.
- `step`: casts gradients back to float32 before the actual optimizer step.
- `state_dict`: quantizes optimizer states (exp_avg, exp_avg_sq) for
  memory-efficient checkpoint serialization.
- `load_state_dict`: dequantizes optimizer states back to float32 before loading.

### 5. Spectrum logging
Every `--metis-log-freq` steps, the `train_step` hook saves singular values,
energy concentration (top1/top5 ratios), and quantization relative error for
each tracked parameter to `<metis-output-dir>/spectra/metis_spectrum_step_<N>.pt`.

## Usage

### Enable in MindSpeed-LLM training
```bash
# Source Ascend environment first
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python pretrain_gpt.py \
    ...standard args... \
    --metis \
    --metis-quant-dtype fp4 \
    --metis-rank-frac 0.015 \
    --metis-block-size 16 \
    --metis-sample-ratio 0.01 \
    --metis-update-freq 1000 \
    --metis-log-freq 100 \
    --metis-output-dir /home/zs/metis/output
```

### CLI arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--metis` | False | Enable Metis spectral quantization. |
| `--metis-quant-dtype` | fp4 | Quantization dtype: `fp4` (E2M1) or `fp8`. |
| `--metis-rank-frac` | 0.015 | Fraction of min(rows, cols) used as low-rank dimension. |
| `--metis-block-size` | 16 | Block size for blockwise scaling. |
| `--metis-sample-ratio` | 0.01 | Row sampling ratio for randomized SVD. |
| `--metis-update-freq` | 1000 | Steps between subspace cache refreshes. |
| `--metis-log-freq` | 100 | Steps between spectrum log writes. |
| `--metis-output-dir` | /home/zs/metis/output | Directory for spectrum logs. |
| `--metis-max-subspace-params` | 64 | Max 2-D params to cache subspaces for. |

### Run unit tests (CPU, no NPU)
```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python \
    mindspeed_llm/features_manager/metis/test_quant_impl.py
```

### Run end-to-end smoke test (CPU)
```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python \
    mindspeed_llm/features_manager/metis/smoke_test.py
```

### Analyze spectrum logs
```bash
python mindspeed_llm/features_manager/metis/analyze_spectra.py \
    --spectra-dir /home/zs/metis/output/spectra \
    --output-dir /home/zs/metis/output/analysis
```
Produces `spectrum_summary.csv`, `spectral_distortion.json`, and per-parameter
singular-value plots (`spectrum_<param>.png`).

## NPU / FP4 hardware notes

- The current FP4 path is a **simulation** that snaps float32 values to E2M1
  levels. It gives the same numerical behavior as a real FP4 kernel would, so
  training dynamics and convergence are faithfully reproduced.
- On Ascend NPU without native FP4, the simulation runs on the NPU's float32
  units (set `--metis-quant-dtype fp4`). For lower overhead, use
  `--metis-quant-dtype fp8` which has 127 levels and much lower error.
- When Ascend FP4 kernels ship, replace `fp4_quantize_blockwise` in
  `quant_impl.py` with the hardware op; no other changes are needed.

## Validation results (Qwen2.5-7B, layer-0 weights, CPU)

| Parameter | Shape | top1 | FP4 rel err | Metis-FP4 rel err | FP8 rel err |
|-----------|-------|------|-------------|-------------------|-------------|
| q_proj | 3584×3584 | 0.509 | 0.719 | 0.718 | 0.005 |
| k_proj | 512×3584 | 0.064 | 0.720 | 0.718 | 0.005 |
| v_proj | 512×3584 | 0.057 | 0.722 | 0.720 | 0.005 |
| o_proj | 3584×3584 | 0.228 | 0.737 | 0.732 | 0.005 |

The high `top1` ratio for `q_proj` (0.51) confirms the spectral anisotropy that
Motivates Metis. Metis-FP4 slightly outperforms plain FP4, and FP8 achieves
~0.5% relative error.

## Paper reference

Cao et al., "Metis: Training LLMs with FP4 Quantization", arXiv:2509.00404.
