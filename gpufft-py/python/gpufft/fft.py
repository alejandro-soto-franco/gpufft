"""gpufft.fft submodule.

The native FFT functions live on the ``fft`` submodule inside the ``_native``
PyO3 cdylib. PyO3 submodules are not importable as on-disk packages, so this
wrapper pulls them via attribute access.

The ``fft_1d_c2c_pow2`` signature matches ``ferrum_gpu.fft.fft_1d_c2c_pow2``
so code written against one package runs on the other; gpufft adds a
``backend=`` selector ("cuda" or "vulkan") and ``fft_1d_c2c`` for arbitrary N.
"""

from gpufft._native import fft as _fft_native

fft_1d_c2c_pow2 = _fft_native.fft_1d_c2c_pow2
fft_1d_c2c = _fft_native.fft_1d_c2c

__all__ = ["fft_1d_c2c_pow2", "fft_1d_c2c"]
