# Changelog

All notable changes to `gpufft` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows semantic versioning.

## [0.1.3] - 2026-05-26

### Fixed
- `VulkanC2cPlan::execute` and `VulkanC2cPlan::execute_shared` with
  `PlanDesc::normalize = true` produced incorrect normalization on the
  second and later calls when reused across different `Buffer<Complex32>`
  instances. Root cause: VkFFT's `VkFFTCheckUpdateBufferSet` decides
  whether to rebuild its descriptor by comparing
  `launchParams.buffer` against `app->configuration.buffer` as raw
  *pointer addresses*, not by dereferencing and comparing the
  `VkBuffer` values, so a stack-local slot reused across calls
  silently skipped the rebuild and the descriptor stayed bound to the
  previous buffer. Fix: store two heap-stable `u64` slots on the plan's
  boxed `Inner` and toggle between them whenever the caller's `VkBuffer`
  raw handle changes, forcing VkFFT to observe a different pointer and
  rebuild the descriptor; same-buffer repeats keep the same slot and
  cost only a single `u64` equality check. `VulkanR2cPlan` and
  `VulkanC2rPlan` were not affected because they always submit VkFFT
  against the plan's internal `fft_buffer` (heap-stable, never changes).
