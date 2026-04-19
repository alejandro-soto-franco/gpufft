# gpufft

Unified GPU-accelerated FFT for Rust, backed by **VkFFT** on Vulkan and **cuFFT** on CUDA.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

> Status: pre-release (0.1). Vulkan backend is functional; CUDA backend is a stub.

## Why

The Rust ecosystem has good CPU FFT libraries (`rustfft`, `ndrustfft`) and solid
graphics/compute frameworks (`wgpu`, `cudarc`, `cubecl`), but no ergonomic
cross-backend GPU FFT. `gpufft` fills that gap with a single trait surface that
works the same on NVIDIA, AMD, and Intel GPUs.

- **Vulkan backend** wraps [VkFFT](https://github.com/DTolm/VkFFT) via a thin FFI
  shim, with automatic buffer interop to `wgpu` via `create_buffer_from_hal`.
- **CUDA backend** wraps [cuFFT](https://docs.nvidia.com/cuda/cufft/) via
  [cudarc](https://crates.io/crates/cudarc). *(Implementation in progress.)*
- Backends are selected by Cargo feature flags at build time. Buffers and plans
  are typed at the backend level, so cross-backend misuse is a compile error.

## Non-goals (v0.1)

- **Cross-backend buffer sharing.** External-memory interop is a future concern.
- **General GPU compute framework.** Use [`wgpu`](https://crates.io/crates/wgpu)
  or [`cubecl`](https://crates.io/crates/cubecl) for that; `gpufft` is focused
  on FFT.
- **Real-to-real transforms (DCT/DST), 4D+ shapes, non-power-of-two auto-tuning.**
  These may appear in later releases as demand dictates.

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
gpufft = { version = "0.1", features = ["vulkan"] }
```

Enable both backends simultaneously with `features = ["vulkan", "cuda"]`.

### System prerequisites

**Vulkan backend (Fedora):**

```sh
sudo dnf install vulkan-headers vulkan-loader-devel \
                 glslang-devel spirv-tools-devel
```

**CUDA backend:** CUDA Toolkit 12.5 or later on the build host. Runtime requires
a matching NVIDIA driver.

## Usage

Each device exposes three plan types:

| Method                | Use                                 |
|-----------------------|-------------------------------------|
| `plan_c2c::<T>()`     | complex-to-complex, in-place        |
| `plan_r2c::<F>()`     | real-to-complex, forward only       |
| `plan_c2r::<F>()`     | complex-to-real, inverse only       |

```rust
use gpufft::{
    vulkan::VulkanBackend, BufferOps, C2cPlanOps, Device, Direction, PlanDesc, Shape,
};
use num_complex::Complex32;

let device = VulkanBackend::new_device(Default::default())?;
let mut input = device.alloc::<Complex32>(1024)?;
input.write(&host_data)?;

let mut plan = device.plan_c2c::<Complex32>(&PlanDesc {
    shape: Shape::D1(1024),
    batch: 1,
    normalize: false,
})?;
plan.execute(&mut input, Direction::Forward)?;

let mut host_out = vec![Complex32::default(); 1024];
input.read(&mut host_out)?;
```

Backend-generic code works identically against any backend:

```rust
use gpufft::{Backend, C2cPlanOps, Direction};
use num_complex::Complex32;

fn forward_c2c<B: Backend>(
    plan: &mut B::C2cPlan<Complex32>,
    buf: &mut B::Buffer<Complex32>,
) -> Result<(), B::Error> {
    plan.execute(buf, Direction::Forward)
}
```

R2C (forward real-to-complex) and C2R (inverse complex-to-real) use
Hermitian-symmetric half-spectra on the last dimension (`n / 2 + 1`), matching
VkFFT and cuFFT conventions:

```rust
use gpufft::{BufferOps, Device, PlanDesc, R2cPlanOps, Shape};

let mut real = device.alloc::<f32>(1024)?;
let mut spectrum = device.alloc::<num_complex::Complex32>(513)?; // 1024/2 + 1

let mut r2c = device.plan_r2c::<f32>(&PlanDesc {
    shape: Shape::D1(1024),
    batch: 1,
    normalize: false,
})?;
r2c.execute(&real, &mut spectrum)?;
```

## Crate layout

| Crate                | Purpose                                                   |
|----------------------|-----------------------------------------------------------|
| `gpufft`             | Public API, trait surface, backend modules                |
| `gpufft-vulkan-sys`  | FFI bindings to vendored VkFFT                            |
| `gpufft-cuda-sys`    | FFI bindings to cuFFT (system CUDA Toolkit)               |

## License

Apache-2.0. See [LICENSE-APACHE](LICENSE-APACHE) and [NOTICE](NOTICE).
