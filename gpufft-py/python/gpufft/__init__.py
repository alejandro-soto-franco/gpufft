"""gpufft: cross-vendor GPU FFT for Python (cuFFT + VkFFT).

The same FFT call runs on NVIDIA via cuFFT (``backend="cuda"``) or on any
Vulkan device via VkFFT (``backend="vulkan"``: AMD, Intel, Apple, NVIDIA).
The ``fft`` surface mirrors ``ferrum_gpu`` so the two packages are swappable.
"""

from gpufft._native import available_backends, version
from gpufft import cuda, fft, vulkan

__all__ = ["version", "available_backends", "cuda", "vulkan", "fft"]
