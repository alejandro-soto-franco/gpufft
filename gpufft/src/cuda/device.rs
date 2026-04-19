//! CUDA device construction and shared context.

use std::sync::Arc;

use gpufft_cuda_sys as sys;

use super::buffer::CudaBuffer;
use super::error::{CudaError, check_cuda};
use super::plan::{CudaC2cPlan, CudaC2rPlan, CudaR2cPlan};
use crate::backend::Device;
use crate::plan::PlanDesc;
use crate::scalar::{Complex, Real, Scalar};

/// Options controlling [`CudaDevice`] construction.
#[derive(Clone, Debug, Default)]
pub struct DeviceOptions {
    /// Ordinal of the CUDA device to use. Defaults to 0.
    pub device_ordinal: Option<i32>,
}

/// Shared CUDA state held by [`CudaDevice`], [`CudaBuffer`], and the plan
/// types via [`Arc`]. Resources can then be cleaned up in any drop order.
pub(crate) struct CudaContext {
    pub(crate) device_ordinal: i32,
}

impl CudaContext {
    /// Bind the calling thread to this context's device. Every public-API
    /// entry point calls this before touching CUDA state so switching
    /// between multiple `CudaDevice` instances (different ordinals) is safe.
    pub(crate) fn make_current(&self) -> Result<(), CudaError> {
        // SAFETY: `cudaSetDevice` is a thread-local switch; no pointers escape.
        unsafe { check_cuda("cudaSetDevice", sys::cudaSetDevice(self.device_ordinal)) }
    }
}

/// A CUDA compute device bound to a single physical GPU.
pub struct CudaDevice {
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaDevice {
    /// Construct a new device from the given options.
    pub fn new(options: DeviceOptions) -> Result<Self, CudaError> {
        let mut count: i32 = 0;
        // SAFETY: `cudaGetDeviceCount` writes through the provided int pointer.
        unsafe {
            check_cuda("cudaGetDeviceCount", sys::cudaGetDeviceCount(&mut count))?;
        }
        if count == 0 {
            return Err(CudaError::NoDevice);
        }

        let ordinal = options.device_ordinal.unwrap_or(0);
        if ordinal < 0 || ordinal >= count {
            return Err(CudaError::DeviceOutOfRange {
                requested: ordinal,
                count,
            });
        }

        // SAFETY: ordinal is within range validated above.
        unsafe {
            check_cuda("cudaSetDevice", sys::cudaSetDevice(ordinal))?;
        }

        Ok(Self {
            ctx: Arc::new(CudaContext {
                device_ordinal: ordinal,
            }),
        })
    }

    /// Return the ordinal of the selected device.
    pub fn ordinal(&self) -> i32 {
        self.ctx.device_ordinal
    }
}

impl Device<super::CudaBackend> for CudaDevice {
    fn alloc<T: Scalar>(&self, len: usize) -> Result<CudaBuffer<T>, CudaError> {
        CudaBuffer::new(self.ctx.clone(), len)
    }

    fn plan_c2c<T: Complex>(&self, desc: &PlanDesc) -> Result<CudaC2cPlan<T>, CudaError> {
        CudaC2cPlan::new(self.ctx.clone(), *desc)
    }

    fn plan_r2c<F: Real>(&self, desc: &PlanDesc) -> Result<CudaR2cPlan<F>, CudaError> {
        CudaR2cPlan::new(self.ctx.clone(), *desc)
    }

    fn plan_c2r<F: Real>(&self, desc: &PlanDesc) -> Result<CudaC2rPlan<F>, CudaError> {
        CudaC2rPlan::new(self.ctx.clone(), *desc)
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx.make_current()?;
        // SAFETY: `cudaDeviceSynchronize` takes no arguments.
        unsafe {
            check_cuda("cudaDeviceSynchronize", sys::cudaDeviceSynchronize())?;
        }
        Ok(())
    }
}
