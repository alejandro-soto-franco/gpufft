# VkFFT multi-dim C2R + `isInputFormatted` issue

Standalone C++ repro showing that a **dedicated C2R (complex-to-real
inverse)** plan configured with `performR2C = 1`, `isInputFormatted = 1`,
`isOutputFormatted = 1` and launch-time buffer overrides produces
incorrect output on multi-dimensional shapes.

## Summary

| Mode | What                                                                       | 1D | 2D | 3D |
|------|----------------------------------------------------------------------------|----|----|----|
| A    | `isInputFormatted = 0`, padded in-place R2C (baseline)                     | ✓  | ✓  | ✓  |
| B    | R2C, `isInputFormatted = 1`, user buffers at init                          | ✓  | ✓  | ✓  |
| C    | R2C, `isInputFormatted = 1`, scratch at init, user via `VkFFTLaunchParams` | ✓  | ✓  | ✓  |
| D    | **Round-trip** R2C then C2R, two plans, launch-param overrides             | ✓  | ✗  | ✗  |

Modes A / B / C all produce bit-identical R2C spectra (L∞ = 0.0 exactly).
Mode D's round-trip is `c2r(r2c(x))`, which should equal `x * N` for the
unnormalised convention; the R2C half has already been validated by
modes A/B/C so the error is in the C2R path.

## Measured

On NVIDIA RTX 5060 Laptop GPU, Vulkan 1.4, VkFFT 1.3.4:

```
-- Shape 1024x1x1 (Nx innermost) --
  round-trip R2C+C2R (D): L-inf(back/N - input) = 8.344650e-07  [OK]
-- Shape 64x64x1 (Nx innermost) --
  round-trip R2C+C2R (D): L-inf(back/N - input) = 2.315226e+00  [DIFFER]
-- Shape 32x32x32 (Nx innermost) --
  round-trip R2C+C2R (D): L-inf(back/N - input) = 1.963728e+00  [DIFFER]
-- Shape 16x32x64 (Nx innermost) --
  round-trip R2C+C2R (D): L-inf(back/N - input) = 3.024768e+00  [DIFFER]
```

## C2R config under test (mode D)

A dedicated C2R plan:

```cpp
VkFFTConfiguration cfg{};
cfg.FFTdim = 3;
cfg.size[0] = Nx; cfg.size[1] = Ny; cfg.size[2] = Nz;
cfg.performR2C          = 1;
cfg.isInputFormatted    = 1;
cfg.isOutputFormatted   = 1;
cfg.outputBufferStride[0] = Nx;
cfg.outputBufferStride[1] = Nx * Ny;
cfg.outputBufferStride[2] = Nx * Ny * Nz;
cfg.inputBuffer      = &scratch_in.buf;  // scratch, overridden per-launch
cfg.inputBufferSize  = &complex_bytes; cfg.inputBufferNum = 1;
cfg.buffer           = &scratch_mid.buf; // scratch, overridden per-launch
cfg.bufferSize       = &complex_bytes; cfg.bufferNum = 1;
cfg.outputBuffer     = &scratch_out.buf; // scratch, overridden per-launch
cfg.outputBufferSize = &real_bytes; cfg.outputBufferNum = 1;
initializeVkFFT(&app, cfg);

// Per-launch:
VkFFTLaunchParams lp{};
lp.commandBuffer = &cb;
lp.inputBuffer   = &user_complex.buf;
lp.buffer        = &intermediate.buf;
lp.outputBuffer  = &user_real.buf;
VkFFTAppend(&app, /*inverse=*/1, &lp);
```

The same launch-override pattern works for R2C in the same repro
(mode C), so the issue is not with `VkFFTLaunchParams` override in
general — it is specific to the C2R inverse direction when that plan is
a standalone VkFFT application (not the same app used for both R2C and
C2R with `inverseReturnToInputBuffer = 1`, as in
`sample_15_precision_VkFFT_single_r2c.cpp`).

## Hypotheses (not yet confirmed)

1. `inverseReturnToInputBuffer = 1` may be required for multi-dim C2R
   with `isInputFormatted + isOutputFormatted` when the plan is
   initialised with scratch buffers that get overridden at launch.
2. VkFFT may assume the forward and inverse halves of an R2C/C2R
   round-trip share one application, and the "three-buffer out-of-place"
   path documented for C2R may need a matching R2C sibling in the same
   app.
3. Something about the axis-upload scheduling for multi-dim C2R reads
   the wrong stride when launch-time override meets dedicated C2R
   init.

## Build + run

```
make         # produces ./repro
./repro
```

Requires the same system packages as `gpufft-vulkan-sys`:

- Fedora: `vulkan-headers vulkan-loader-devel glslang-devel spirv-tools-devel`
- Debian/Ubuntu: LunarG Vulkan SDK.

The Makefile expects the vendored VkFFT under
`../../gpufft-vulkan-sys/vendor/VkFFT`. Override with `make VKFFT_ROOT=...`.

## What this means for `gpufft`

The R2C-only zero-copy pattern (mode C) works and could be used in the
Rust backend. But the C2R half of the round-trip fails in C++ as well,
so zero-copy C2R via this configuration cannot be ported until the
upstream issue is diagnosed. The `gpufft` Vulkan backend therefore
retains the compute-shader padder for both R2C (input padding) and
C2R (output strip), keeping the same correctness and performance that
shipped in commit `b0727de`.

Suitable for filing as a VkFFT issue (DTolm/VkFFT) with the repro
binary attached or the `repro.cpp` source inlined.
