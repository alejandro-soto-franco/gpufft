# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Workspace scaffold with three crates: `gpufft`, `gpufft-vulkan-sys`, `gpufft-cuda-sys`.
- Vulkan backend via VkFFT v1.3.4:
  - C2C in-place at 1D/2D/3D, Complex32 and Complex64.
  - R2C forward and C2R inverse at 1D/2D/3D, f32 and f64. Hermitian
    half-spectrum on the contiguous (last, in ndarray row-major) axis,
    matching cuFFT convention.
- CUDA backend via cuFFT (system-linked):
  - C2C in-place, R2C forward, C2R inverse at 1D/2D/3D.
  - f32 (CUFFT_{C2C,R2C,C2R}) and f64 (CUFFT_{Z2Z,D2Z,Z2D}).
- Backend-generic trait surface (`Backend`, `Device`, `BufferOps`,
  `C2cPlanOps`, `R2cPlanOps`, `C2rPlanOps`) with `Real` / `Complex`
  supertraits pairing each scalar with its partner type.
- Integration tests for C2C roundtrip (1D/2D/3D) on both backends, CUDA
  R2C+C2R, and Vulkan R2C+C2R (including a non-cubic shape).
- `r2c_c2r` benchmark comparing Vulkan vs CUDA at vonkarman-typical sizes.

### Fixed
- Descriptor-set crash on `VkFFTAppend`: the VkFFT configuration pointed to
  stack-local `u64` handle copies that went out of scope before the first
  launch. Handle storage is now owned by the boxed plan, so addresses are
  stable for the plan's lifetime.
