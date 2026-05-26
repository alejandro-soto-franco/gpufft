# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-05-26

### Added
- `shared` feature (Linux + `vulkan` + `cuda`) for zero-copy Vulkan↔CUDA
  interop. New module `gpufft::shared` exports `SharedMemory` (host-visible
  mirror) and `SharedFftBuffer` (device-local FFT-ready memory) backed by
  `VK_KHR_external_memory_fd` + `cudaImportExternalMemory`.
- `VulkanC2cPlan::execute_shared` and `CudaC2cPlan::execute_shared`
  inherent methods that run their respective FFT on a `SharedFftBuffer`.
- `VulkanDevice::raw_handles` and `CudaDevice::raw_handles` accessors
  exposing the underlying handles required for cross-backend interop.
- `gpufft-cuda-sys`: bindgen allowlist extended for `cudaImportExternalMemory`,
  `cudaExternalMemoryGetMappedBuffer`, `cudaDestroyExternalMemory`, and the
  supporting handle/buffer descriptor types.

### Changed
- `VulkanDevice::new` now enables `VK_KHR_external_memory` +
  `VK_KHR_external_memory_fd` under `cfg(all(feature = "cuda", target_os = "linux"))`.
  No effect on non-shared builds.

### Fixed
- `VulkanC2cPlan::execute` and `VulkanC2cPlan::execute_shared` with
  `PlanDesc::normalize = true` produced incorrect normalization on the
  second and later calls when reused across different `Buffer<Complex32>`
  instances. Root cause: VkFFT's `VkFFTCheckUpdateBufferSet` decides whether
  to rebuild its descriptor by comparing `launchParams.buffer` against
  `app->configuration.buffer` as raw *pointer addresses*, not by
  dereferencing and comparing the `VkBuffer` values, so a stack-local slot
  reused across calls silently skipped the rebuild and the descriptor stayed
  bound to the previous buffer. Fix: store two heap-stable `u64` slots on
  the plan's boxed `Inner` and toggle between them whenever the caller's
  `VkBuffer` raw handle changes, forcing VkFFT to observe a different
  pointer and rebuild the descriptor; same-buffer repeats keep the same slot
  and cost only a single `u64` equality check. `VulkanR2cPlan` and
  `VulkanC2rPlan` were not affected because they always submit VkFFT against
  the plan's internal `fft_buffer` (heap-stable, never changes).

## [0.1.2] - 2026-04-20

### Added
- Package-level `README.md` in each published crate (`gpufft`,
  `gpufft-vulkan-sys`, `gpufft-cuda-sys`) so crates.io renders a description
  page. Subcrate READMEs clarify that the `*-sys` crates are internal
  plumbing and point users at `gpufft`.

No API changes; patch release for crates.io metadata only.

## [0.1.1] - 2026-04-19

### Changed
- README overhauled to reflect the shipped v0.1 state: CUDA backend is real
  (cuFFT via bindgen on `gpufft-cuda-sys`, not the earlier cudarc stub),
  Vulkan backend is pure-`ash` (no wgpu dependency), compute-shader padder
  lands in `shaders/stride_copy.comp`, R2C / C2R convention documented,
  benchmark table added, design notes and the VkFFT C2R investigation
  referenced.

No API changes; republishing for the accurate metadata.

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
- Stride-copy compute shader (`shaders/stride_copy.comp`, SPIR-V compiled at
  build time via `glslangValidator`) replacing the R2C / C2R multi-region
  `vkCmdCopyBuffer` padding step with a single compute dispatch. Benchmark
  wins on NVIDIA: 32³ 3×, 64³ 6.9×, 128³ 7.5×, 256³ 2.8×. Vulkan now runs
  2–3× slower than cuFFT (down from 10–24×).

### Fixed
- Descriptor-set crash on `VkFFTAppend`: the VkFFT configuration pointed to
  stack-local `u64` handle copies that went out of scope before the first
  launch. Handle storage is now owned by the boxed plan, so addresses are
  stable for the plan's lifetime.
