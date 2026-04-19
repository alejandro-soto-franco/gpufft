//! Raw FFI bindings to the CUDA Runtime + cuFFT.
//!
//! This crate is internal plumbing for the `gpufft` crate. It exposes the
//! minimal subset of the CUDA Runtime needed for device memory and stream
//! management, plus the cuFFT plan creation, execution, and destruction
//! entry points. Downstream crates should depend on `gpufft` instead.
//!
//! # Safety
//!
//! Every item in this crate is `unsafe` by construction. Lifetimes of the
//! CUDA handles (streams, device pointers, cuFFT plans) are the caller's
//! responsibility.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(missing_docs)]
// bindgen emits unsafe blocks for every C call with wrap_unsafe_ops(true);
// the safety contract is documented on the wrappers in the parent crate.
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::missing_safety_doc)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Returns the cuFFT runtime version as a packed integer.
///
/// Encoding is cuFFT's own: `major * 1000 + minor * 10 + patch`.
pub fn cufft_version() -> i32 {
    let mut v: i32 = 0;
    // SAFETY: `cufftGetVersion` writes an int through the provided pointer.
    unsafe {
        cufftGetVersion(&mut v);
    }
    v
}

#[cfg(test)]
mod sanity {
    use super::*;

    #[test]
    fn cufft_linked_and_version_readable() {
        let v = cufft_version();
        assert!(v >= 10000, "cuFFT version suspiciously low: {v}");
    }
}
