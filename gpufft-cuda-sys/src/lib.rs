//! Raw FFI bindings to cuFFT.
//!
//! Stub crate for gpufft v0.1. The CUDA backend in [`gpufft`] compiles
//! against this crate under `--features cuda` but currently returns an
//! `unimplemented` error from every fallible operation. A future release
//! will wire cuFFT via [`cudarc`](https://crates.io/crates/cudarc).

#![warn(missing_docs)]

/// Placeholder version identifier. Replaced once cuFFT linkage is wired up.
pub const CUFFT_STUB: &str = "stub-v0.1";
