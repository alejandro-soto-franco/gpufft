# gpufft-vulkan-sys

Raw FFI bindings to [VkFFT](https://github.com/DTolm/VkFFT), vendored and built
from source at crate-build time. This is internal plumbing for
[`gpufft`](https://crates.io/crates/gpufft); you almost certainly want that
crate, not this one.

[![crates.io](https://img.shields.io/crates/v/gpufft-vulkan-sys.svg)](https://crates.io/crates/gpufft-vulkan-sys)
[![docs.rs](https://img.shields.io/docsrs/gpufft-vulkan-sys)](https://docs.rs/gpufft-vulkan-sys)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alejandro-soto-franco/gpufft/blob/main/LICENSE-APACHE)

## What this crate is

- Vendored copy of VkFFT v1.3.4 (header-only C++, MIT-licensed) under
  `vendor/VkFFT/`.
- A small C shim (`vkfft_shim.c`) that exposes a C ABI to Rust.
- `bindgen`-generated bindings for that shim.
- A `cc` build step that compiles the shim + VkFFT.

## What this crate is not

- A safe Rust API. All symbols are `unsafe`. Use `gpufft` for the typed
  trait surface.
- A Vulkan loader. This crate expects Vulkan headers and loader to be
  available at build time; it does not vendor the Vulkan SDK.

## Build prerequisites

- Vulkan headers + loader: `vulkan-headers`, `vulkan-loader-devel` (Fedora)
  or the [LunarG Vulkan SDK](https://vulkan.lunarg.com/) (Ubuntu/Debian/macOS).
- `glslang` + `SPIRV-Tools` development packages for VkFFT's SPIR-V
  emission path.
- A C++17-capable compiler (invoked via `cc`).

`pkg-config` is used to locate Vulkan. If `pkg-config` fails, set
`VULKAN_SDK` to a directory containing `include/vulkan/` and a loader
library.

## Licensing

`gpufft-vulkan-sys` itself is Apache-2.0. The vendored VkFFT sources are
MIT-licensed; see `vendor/VkFFT/LICENSE` in the packaged crate and the
project `NOTICE` file in the repository.

Repository: <https://github.com/alejandro-soto-franco/gpufft>
