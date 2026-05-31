"""gpufft.vulkan submodule: the VkFFT-backed persistent ``Device`` handle.

PyO3 submodules are not importable as on-disk packages, so this wrapper pulls
the native ``Device`` class via attribute access on the ``_native`` cdylib.
"""

from gpufft._native import vulkan as _vulkan_native

Device = _vulkan_native.Device

__all__ = ["Device"]
