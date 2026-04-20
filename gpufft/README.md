# gpufft

Unified GPU-accelerated FFT for Rust, backed by **VkFFT** on Vulkan and **cuFFT** on CUDA.

[![crates.io](https://img.shields.io/crates/v/gpufft.svg)](https://crates.io/crates/gpufft)
[![docs.rs](https://img.shields.io/docsrs/gpufft)](https://docs.rs/gpufft)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alejandro-soto-franco/gpufft/blob/main/LICENSE-APACHE)

## Why

The Rust ecosystem has good CPU FFT libraries (`rustfft`, `ndrustfft`) and solid
GPU compute frameworks (`wgpu`, `cubecl`, `cudarc`), but no ergonomic
cross-vendor GPU FFT. `gpufft` fills that gap with a single trait surface that
works the same on NVIDIA, AMD, and Intel GPUs.

- **Vulkan backend** wraps [VkFFT](https://github.com/DTolm/VkFFT) via a thin
  FFI shim (`gpufft-vulkan-sys`), on pure `ash` with no `wgpu` dependency.
- **CUDA backend** wraps [cuFFT](https://docs.nvidia.com/cuda/cufft/) via
  bindgen on the system CUDA Toolkit (`gpufft-cuda-sys`).
- Backends are selected by Cargo feature flags. Buffers and plans are typed
  at the **backend plus scalar** level, so mixing a Vulkan buffer with a CUDA
  plan, or an `f32` plan with a `Complex64` buffer, is a compile error.
- One plan-creation method per transform kind: `plan_c2c`, `plan_r2c`,
  `plan_c2r`. f32 (`Complex32` / `f32`) and f64 (`Complex64` / `f64`) at 1D,
  2D, and 3D.

## Installation

```toml
[dependencies]
gpufft = { version = "0.1", features = ["vulkan"] }
# or
gpufft = { version = "0.1", features = ["cuda"] }
# or both
gpufft = { version = "0.1", features = ["vulkan", "cuda"] }
```

### System prerequisites

**Vulkan backend**

- Fedora: `sudo dnf install vulkan-headers vulkan-loader-devel glslang-devel spirv-tools-devel`
- Debian/Ubuntu: install the [LunarG Vulkan SDK](https://vulkan.lunarg.com/).

**CUDA backend**: CUDA Toolkit 12.x or later on the build host (the bindgen
pass needs `cufft.h` + `cuda_runtime.h`). Runtime needs a matching NVIDIA
driver. `CUDA_PATH` / `CUDA_HOME` override the default `/usr/local/cuda` lookup.

## Usage

Each device exposes three plan types:

| Method              | Use                                   |
|---------------------|---------------------------------------|
| `plan_c2c::<T>()`   | complex-to-complex, in-place          |
| `plan_r2c::<F>()`   | real-to-complex, forward only         |
| `plan_c2r::<F>()`   | complex-to-real, inverse only         |

```rust
use gpufft::{
    vulkan::VulkanBackend, BufferOps, C2cPlanOps, Device, Direction, PlanDesc, Shape,
};
use num_complex::Complex32;

let device = VulkanBackend::new_device(Default::default())?;
let mut buffer = device.alloc::<Complex32>(1024)?;
buffer.write(&host_data)?;

let mut plan = device.plan_c2c::<Complex32>(&PlanDesc {
    shape: Shape::D1(1024),
    batch: 1,
    normalize: false,
})?;
plan.execute(&mut buffer, Direction::Forward)?;

let mut host_out = vec![Complex32::default(); 1024];
buffer.read(&mut host_out)?;
```

R2C / C2R use Hermitian-symmetric half-spectra on the last (contiguous)
dimension, matching cuFFT and VkFFT conventions. For `Shape::D3([nx, ny, nz])`
the complex side has `nx * ny * (nz / 2 + 1)` elements.

Backend-generic code composes over any `B: Backend`:

```rust
use gpufft::{Backend, C2cPlanOps, Direction};

fn forward_c2c<B: Backend>(
    plan: &mut B::C2cPlan<num_complex::Complex32>,
    buf: &mut B::Buffer<num_complex::Complex32>,
) -> Result<(), B::Error> {
    plan.execute(buf, Direction::Forward)
}
```

## Performance

3D R2C+C2R pair, f32, 10 iterations after 3 warmup. NVIDIA RTX 5060 Laptop GPU,
Vulkan 1.4, VkFFT 1.3.4, CUDA 13.0 / driver 580:

| Shape | cuFFT    | VkFFT (gpufft)  | ratio |
|-------|----------|-----------------|-------|
| 32³   | 25 µs    | 86 µs           | 3.4×  |
| 64³   | 36 µs    | 105 µs          | 2.9×  |
| 128³  | 117 µs   | 386 µs          | 3.3×  |
| 256³  | 2.5 ms   | 5.28 ms         | 2.1×  |

cuFFT is NVIDIA's vendor-tuned path; VkFFT is the cross-vendor fallback and
is the only option on AMD / Intel GPUs. The Vulkan backend uses a
compute-shader padder to collapse the innermost-axis stride handling for
R2C / C2R into a single dispatch.

## Design notes

- **C2C** is truly zero-copy: the user's buffer is passed through
  `VkFFTLaunchParams.buffer` at dispatch time.
- **R2C / C2R** use a compute-shader padder to align the real innermost
  axis to VkFFT's `2 * (n/2 + 1)` stride.
- **Plan lifetimes**: `VkFFTConfiguration` retains raw pointers to its
  handle fields for the application's lifetime, so all handle storage
  lives inside a boxed `Inner` struct with a stable heap address.

## Non-goals (v0.1)

- Cross-backend buffer sharing (external-memory interop is a later concern).
- General GPU compute framework; use `wgpu` / `cubecl` for that.
- Real-to-real transforms (DCT / DST), 4D+ shapes, non-power-of-two
  auto-tuning.

## Crate layout

| Crate                | Purpose                                             |
|----------------------|-----------------------------------------------------|
| `gpufft`             | Public API, trait surface, backend modules         |
| `gpufft-vulkan-sys`  | FFI bindings to vendored VkFFT                     |
| `gpufft-cuda-sys`    | FFI bindings to cuFFT (system CUDA Toolkit)        |

Repository: <https://github.com/alejandro-soto-franco/gpufft>

## License

Apache-2.0.
