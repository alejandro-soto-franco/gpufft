//! CUDA backend (cuFFT).
//!
//! Wraps the CUDA Runtime for device memory and stream management, and
//! cuFFT for FFT plan creation and execution. Precision is derived from
//! the scalar type: `Complex32` + `f32` use CUFFT_C2C / CUFFT_R2C /
//! CUFFT_C2R; `Complex64` + `f64` use CUFFT_Z2Z / CUFFT_D2Z / CUFFT_Z2D.
//!
//! Limitations in this release:
//!
//! - `PlanDesc::normalize = true` is rejected with
//!   [`CudaError::UnsupportedNormalize`]. cuFFT does not normalise
//!   inverses natively; callers should scale in host code or select the
//!   Vulkan backend (which delegates normalisation to VkFFT). This will
//!   be wired in a follow-up via a scale kernel.
//! - Batched 2D / 3D plans are not yet exposed (v0.2.0 scope).

use crate::backend::Backend;
use crate::scalar::{Complex, Real, Scalar};

mod buffer;
mod device;
mod error;
mod plan;

pub use buffer::CudaBuffer;
pub use device::{CudaDevice, DeviceOptions};
pub use error::CudaError;
pub use plan::{CudaC2cPlan, CudaC2rPlan, CudaR2cPlan};

/// Marker type implementing [`Backend`] for the cuFFT-backed CUDA backend.
#[derive(Clone, Copy, Debug)]
pub struct CudaBackend;

impl CudaBackend {
    /// Construct a new [`CudaDevice`] using the given options.
    pub fn new_device(options: DeviceOptions) -> Result<CudaDevice, CudaError> {
        CudaDevice::new(options)
    }
}

impl Backend for CudaBackend {
    type Device = CudaDevice;
    type Buffer<T: Scalar> = CudaBuffer<T>;
    type C2cPlan<T: Complex> = CudaC2cPlan<T>;
    type R2cPlan<F: Real> = CudaR2cPlan<F>;
    type C2rPlan<F: Real> = CudaC2rPlan<F>;
    type Error = CudaError;

    const NAME: &'static str = "cuda";
}
