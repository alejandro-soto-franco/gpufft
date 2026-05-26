//! Vulkan↔CUDA zero-copy interop (Linux only, both backends required).
//!
//! See [`SharedMemory`] (host-visible mirror) and [`SharedFftBuffer`]
//! (device-local FFT-ready memory) for the two surface types.
//!
//! Skeleton; SharedMemory + SharedFftBuffer land in Tasks 4 and 5.

#![cfg(all(feature = "vulkan", feature = "cuda", target_os = "linux"))]
