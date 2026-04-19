//! Raw FFI bindings to the vendored VkFFT C++ library.
//!
//! This crate is internal plumbing for the `gpufft` crate. It exposes the
//! minimal subset of VkFFT needed to initialize an FFT application from a
//! Rust-built `VkFFTConfiguration`, append dispatches to a caller-allocated
//! Vulkan command buffer via `VkFFTAppend`, and release resources on drop.
//!
//! # Safety
//!
//! Every item in this crate is `unsafe` by construction. Lifetimes of the
//! Vulkan handles and of the storage backing the `VkFFTConfiguration`
//! pointer fields are the caller's responsibility. Downstream crates should
//! depend on `gpufft` instead, which wraps these calls in a typed API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(missing_docs)]
// bindgen emits unsafe blocks wrapping C calls with `wrap_unsafe_ops(true)`;
// the safety contract for each call lives in the wrapper's own doc comment
// rather than on every generated block.
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::missing_safety_doc)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Pinned VkFFT submodule tag that this crate was built against.
pub const VKFFT_VENDOR_TAG: &str = "v1.3.4";

/// Returns VkFFT's runtime version as a packed integer.
///
/// Encoding follows VkFFT's own convention: `major * 10000 + minor * 100 + patch`.
/// For v1.3.4 this is `10304`.
pub fn vkfft_runtime_version() -> u64 {
    // SAFETY: `gpufft_vkfft_version` takes no arguments and returns a plain int;
    // there are no lifetime or aliasing concerns.
    unsafe { gpufft_vkfft_version() as u64 }
}

#[cfg(test)]
mod sanity {
    use super::*;

    #[test]
    fn vkfft_linked_and_version_readable() {
        let v = vkfft_runtime_version();
        assert!(v >= 10000, "VkFFT version suspiciously low: {v}");
        assert_eq!(VKFFT_VENDOR_TAG, "v1.3.4");
    }
}
