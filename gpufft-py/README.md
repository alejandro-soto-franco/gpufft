# gpufft (Python)

Cross-vendor GPU FFT for Python. One API over two battle-tested libraries:

- **cuFFT** on NVIDIA (`backend="cuda"`)
- **VkFFT** on any Vulkan device (`backend="vulkan"`): AMD, Intel, Apple Silicon (MoltenVK), NVIDIA

```bash
pip install gpufft
```

```python
import numpy as np
import gpufft

x = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)

# NVIDIA via cuFFT
X = gpufft.fft.fft_1d_c2c_pow2(x, log_n=12, backend="cuda")

# Same call on any Vulkan GPU via VkFFT
X = gpufft.fft.fft_1d_c2c_pow2(x, log_n=12, backend="vulkan")

# Arbitrary (non-power-of-two) sizes
y = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
Y = gpufft.fft.fft_1d_c2c(y, n=1000, backend="cuda")
```

The `fft_1d_c2c_pow2` signature matches
[`ferrum-gpu`](https://pypi.org/project/ferrum-gpu/)'s, so code written against
one package runs on the other. `ferrum-gpu` is the pure-Rust kernel path
(transparent, hackable); `gpufft` is the vendor-library path (near-parity
performance, cross-vendor today).

## Persistent device handles

```python
dev = gpufft.cuda.Device(0)        # or gpufft.vulkan.Device(0)
X = gpufft.fft.fft_1d_c2c_pow2(x, log_n=12, device=dev)
```

## Runtime requirements

- NVIDIA path: an NVIDIA driver providing `libcufft.so` (CUDA 13.x).
- Vulkan path: a Vulkan 1.x loader (`libvulkan.so.1`) and an installed ICD.

Neither library is bundled; both are loaded from the system at runtime.

## Status

This release ships **1D complex-to-complex** transforms on both backends.
2D/3D, real transforms (R2C/C2R), and `f64` are on the roadmap.

Apache-2.0.
