//! Error type for the CUDA backend.

use std::ffi::CStr;

use gpufft_cuda_sys as sys;
use thiserror::Error;

/// Errors returned by the CUDA backend.
#[derive(Debug, Error)]
pub enum CudaError {
    /// The system reports no CUDA-capable devices.
    #[error("no CUDA-capable device found")]
    NoDevice,

    /// Requested device index exceeds the number of installed devices.
    #[error("device index {requested} out of range (device count: {count})")]
    DeviceOutOfRange {
        /// Requested ordinal.
        requested: i32,
        /// Number of available devices.
        count: i32,
    },

    /// A CUDA Runtime API call failed.
    #[error("CUDA runtime error: {context}: {message} (code {code})")]
    Runtime {
        /// Human-readable phase ("cudaMalloc", "cudaMemcpy", ...).
        context: &'static str,
        /// The raw CUDA error code.
        code: i32,
        /// Decoded message from `cudaGetErrorString`.
        message: String,
    },

    /// A cuFFT API call failed.
    #[error("cuFFT error: {context}: {code}")]
    CuFft {
        /// Human-readable phase ("cufftPlan3d", "cufftExecC2C", ...).
        context: &'static str,
        /// The raw `cufftResult` code.
        code: u32,
    },

    /// A buffer or slice length did not match the plan's expected extent.
    #[error("length mismatch: expected {expected} elements, got {got}")]
    LengthMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count observed.
        got: usize,
    },

    /// Plan specification was internally inconsistent.
    #[error("invalid plan: {0}")]
    InvalidPlan(&'static str),

    /// `PlanDesc::normalize = true` was requested but CUDA backend does
    /// not yet implement on-device scaling.
    #[error(
        "PlanDesc::normalize = true is not yet implemented in the CUDA \
         backend. Scale on the host after C2R, or use the Vulkan backend."
    )]
    UnsupportedNormalize,
}

impl CudaError {
    pub(crate) fn runtime(context: &'static str, code: sys::cudaError_t) -> Self {
        Self::Runtime {
            context,
            code: code as i32,
            message: cuda_error_string(code),
        }
    }

    pub(crate) fn cufft(context: &'static str, code: sys::cufftResult) -> Self {
        Self::CuFft { context, code }
    }
}

fn cuda_error_string(code: sys::cudaError_t) -> String {
    // SAFETY: `cudaGetErrorString` returns a pointer to a static null-
    // terminated C string or NULL for unknown codes. We guard the NULL
    // case explicitly.
    unsafe {
        let ptr = sys::cudaGetErrorString(code);
        if ptr.is_null() {
            return format!("cuda error {code:?}");
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

pub(crate) fn check_cuda(context: &'static str, code: sys::cudaError_t) -> Result<(), CudaError> {
    if code == sys::cudaError_cudaSuccess {
        Ok(())
    } else {
        Err(CudaError::runtime(context, code))
    }
}

pub(crate) fn check_cufft(context: &'static str, code: sys::cufftResult) -> Result<(), CudaError> {
    if code == sys::cufftResult_t_CUFFT_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::cufft(context, code))
    }
}
