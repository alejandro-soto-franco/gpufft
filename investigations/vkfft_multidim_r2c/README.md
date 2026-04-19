# VkFFT multi-dim R2C + `isInputFormatted` investigation

This directory contains a standalone C++ repro I wrote to file a bug
against VkFFT. The repro **disproved** my bug hypothesis: VkFFT is
correct. My Rust binding had a mistake I never pinned down.

## Hypothesis (wrong)

While writing the Vulkan R2C / C2R plans in the `gpufft` crate, my
"zero-copy" attempt — `isInputFormatted = 1`, user buffers supplied
per-execute via `VkFFTLaunchParams.inputBuffer` / `.buffer` — produced
correct output on 1D shapes but wrong output on 3D (cubic 32³ L∞ ~ 2,
non-cubic 16×32×64 L∞ ~ 3.6). I suspected VkFFT's multi-axis dispatch
path mishandled `isInputFormatted = 1`.

## Experiment

`repro.cpp` runs the **same** 3D R2C forward on the **same** input
through three code paths and compares the complex spectra:

| Mode | Config                                                                   |
|------|--------------------------------------------------------------------------|
| A    | `isInputFormatted = 0`, padded in-place, single buffer. Baseline.        |
| B    | `isInputFormatted = 1`, user buffers bound at `initializeVkFFT` time.    |
| C    | `isInputFormatted = 1`, **scratch** buffers bound at init, real buffers overridden via `VkFFTLaunchParams.inputBuffer` / `.buffer` at execute time. |

Mode C is the pattern a typical plan cache (including `gpufft`) wants:
the plan is built once, before the user's runtime buffers exist, and
actual handles are supplied per call.

## Result on NVIDIA RTX 5060 Laptop GPU, VkFFT 1.3.4

```
-- Shape 1024x1x1 (Nx innermost) --
  padded in-place (A):            ||.||_inf = 350.639465
  formatted, user at init (B):    diff vs A  L-inf = 0.000000, L2_rel = 0.000000e+00  [OK]
  formatted, launch-override (C): diff vs A  L-inf = 0.000000, L2_rel = 0.000000e+00  [OK]

-- Shape 64x64x1 (Nx innermost) --
  padded in-place (A):            ||.||_inf = 1689.841309
  formatted, user at init (B):    diff vs A  L-inf = 0.000000, L2_rel = 0.000000e+00  [OK]
  formatted, launch-override (C): diff vs A  L-inf = 0.000000, L2_rel = 0.000000e+00  [OK]

-- Shape 32x32x32 (Nx innermost) --
  padded in-place (A):            ||.||_inf = 8540.783203
  formatted, user at init (B):    diff vs A  L-inf = 0.000000, L2_rel = 0.000000e+00  [OK]
  formatted, launch-override (C): diff vs A  L-inf = 0.000000, L2_rel = 0.000000e+00  [OK]

-- Shape 16x32x64 (Nx innermost) --
  padded in-place (A):            ||.||_inf = 7486.806152
  formatted, user at init (B):    diff vs A  L-inf = 0.000000, L2_rel = 0.000000e+00  [OK]
  formatted, launch-override (C): diff vs A  L-inf = 0.000000, L2_rel = 0.000000e+00  [OK]
```

All three modes agree **bit-for-bit** (L∞ = 0.0 exactly, not just small
float noise). Including the launch-param-override pattern on multi-dim
shapes.

## What this tells us

1. There is no VkFFT bug to file. The library handles
   `isInputFormatted = 1` correctly for multi-dim R2C.
2. There is no constraint that forbids binding scratch handles at
   `initializeVkFFT` time and overriding via `VkFFTLaunchParams` at
   execute time. That pattern works on 3D.
3. My earlier Rust attempt had a mistake I never pinned down (likely
   in stride derivation, pointer-of-pointer construction on the Rust
   side, or buffer sizing). The current `gpufft` Vulkan backend uses a
   compute-shader padder instead, which is correct and fast enough
   (2–3× slower than cuFFT on NVIDIA).

## Follow-up opportunity

A zero-copy R2C / C2R path in Rust would eliminate the remaining
user→scratch copy that the compute-shader padder still performs.
Expected win: ~10–20% more at 128³+. Since the C++ repro proves the
pattern works, retrying this in Rust is a bounded, actionable task —
just diff the Rust code against `mode_c_r2c` in `repro.cpp` until they
match.

## Build

```
make         # produces ./repro
./repro
```

Requires the same system packages as `gpufft-vulkan-sys`:

- Fedora: `vulkan-headers vulkan-loader-devel glslang-devel spirv-tools-devel`
- Debian/Ubuntu: LunarG Vulkan SDK.

The Makefile expects the vendored VkFFT under
`../../gpufft-vulkan-sys/vendor/VkFFT`. Override with `make VKFFT_ROOT=...`.
