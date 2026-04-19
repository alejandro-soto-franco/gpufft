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

```rust
use gpufft::{
    vulkan::VulkanBackend, BufferOps, Device, Direction, PlanDesc, PlanOps, Shape, Transform,
};
use num_complex::Complex32;

let device = VulkanBackend::new_device(Default::default())?;
let mut input = device.alloc::<Complex32>(1024)?;

input.write(&host_data)?;

let mut plan = device.plan::<Complex32>(&PlanDesc {
    shape: Shape::D1(1024),
    transform: Transform::C2c,
    batch: 1,
    normalize: false,
})?;
plan.execute(&mut input, Direction::Forward)?;

let mut host_out = vec![Complex32::default(); 1024];
input.read(&mut host_out)?;
```

Backend-generic code works identically against both backends:

```rust
use gpufft::{Backend, BufferOps, Direction, PlanOps};

fn forward<B: Backend>(
    plan: &mut B::Plan<num_complex::Complex32>,
    buf: &mut B::Buffer<num_complex::Complex32>,
) -> Result<(), B::Error> {
    plan.execute(buf, Direction::Forward)
}
```

## Crate layout

| Crate                | Purpose                                                   |
|----------------------|-----------------------------------------------------------|
| `gpufft`             | Public API, trait surface, backend modules                |
| `gpufft-vulkan-sys`  | FFI bindings to vendored VkFFT                            |
| `gpufft-cuda-sys`    | FFI bindings to cuFFT (system CUDA Toolkit)               |

## License

Apache-2.0. See [LICENSE-APACHE](LICENSE-APACHE) and [NOTICE](NOTICE).
