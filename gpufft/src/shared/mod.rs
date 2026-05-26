//! Vulkan↔CUDA zero-copy interop (Linux only, both backends required).
//!
//! See [`SharedMemory`] (host-visible mirror) for the zero-copy buffer type.
//! `SharedFftBuffer` (device-local FFT-ready memory) lands in Task 5.

#![cfg(all(feature = "vulkan", feature = "cuda", target_os = "linux"))]

mod memory;
pub use memory::SharedMemory;

use thiserror::Error;

/// Errors produced by the [`shared`](self) module.
///
/// Wraps Vulkan and CUDA failures as formatted strings so the caller does not
/// need to depend on both backend error types simultaneously.
#[derive(Debug, Error)]
pub enum SharedMemoryError {
    /// A Vulkan API call failed during shared-memory setup or teardown.
    #[error("shared-memory interop (Vulkan): {0}")]
    Vulkan(String),

    /// A CUDA Runtime API call failed during shared-memory setup or teardown.
    #[error("shared-memory interop (CUDA): {0}")]
    Cuda(String),
}
