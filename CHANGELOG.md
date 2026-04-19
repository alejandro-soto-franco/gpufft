# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Workspace scaffold with three crates: `gpufft`, `gpufft-vulkan-sys`, `gpufft-cuda-sys`.
- Vulkan backend via VkFFT v1.3.4 (ported from cartan-gpu v0.5).
- CUDA backend stub (compile-only, implementations pending).
- Backend-generic trait surface (`Backend`, `Device`, `Plan`, `Buffer`).
- 1D, 2D, and 3D complex-to-complex FFT roundtrip integration tests.

### Fixed
- Descriptor-set crash on `VkFFTAppend`: the VkFFT configuration pointed to
  stack-local `u64` handle copies that went out of scope before the first
  launch. Handle storage is now owned by the boxed plan, so addresses are
  stable for the plan's lifetime.
