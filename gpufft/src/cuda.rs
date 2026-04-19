//! CUDA backend (cuFFT). **Stub: all operations return an unimplemented error.**
//!
//! The CUDA backend is scheduled for the next release. This module compiles
//! under `--features cuda` so that backend-generic user code can already be
//! written, but every constructor and every fallible method returns
//! [`CudaError::Unimplemented`].

use std::marker::PhantomData;

use thiserror::Error;

use crate::backend::{Backend, BufferOps, C2cPlanOps, C2rPlanOps, Device, R2cPlanOps};
use crate::plan::{Direction, PlanDesc};
use crate::scalar::{Complex, Real, Scalar};

/// Errors returned by the CUDA backend stub.
#[derive(Debug, Error)]
pub enum CudaError {
    /// The CUDA backend is not yet implemented.
    #[error("CUDA backend is not yet implemented (gpufft stub)")]
    Unimplemented,
}

/// Marker type implementing [`Backend`] for the cuFFT-backed CUDA backend.
#[derive(Clone, Copy, Debug)]
pub struct CudaBackend;

impl CudaBackend {
    /// Construct a new [`CudaDevice`]. Always returns [`CudaError::Unimplemented`].
    pub fn new_device() -> Result<CudaDevice, CudaError> {
        Err(CudaError::Unimplemented)
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

/// CUDA device handle (stub).
pub struct CudaDevice {
    _private: (),
}

impl Device<CudaBackend> for CudaDevice {
    fn alloc<T: Scalar>(&self, _len: usize) -> Result<CudaBuffer<T>, CudaError> {
        Err(CudaError::Unimplemented)
    }

    fn plan_c2c<T: Complex>(&self, _desc: &PlanDesc) -> Result<CudaC2cPlan<T>, CudaError> {
        Err(CudaError::Unimplemented)
    }

    fn plan_r2c<F: Real>(&self, _desc: &PlanDesc) -> Result<CudaR2cPlan<F>, CudaError> {
        Err(CudaError::Unimplemented)
    }

    fn plan_c2r<F: Real>(&self, _desc: &PlanDesc) -> Result<CudaC2rPlan<F>, CudaError> {
        Err(CudaError::Unimplemented)
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        Err(CudaError::Unimplemented)
    }
}

/// CUDA buffer (stub).
pub struct CudaBuffer<T: Scalar> {
    _marker: PhantomData<T>,
}

impl<T: Scalar> BufferOps<CudaBackend, T> for CudaBuffer<T> {
    fn len(&self) -> usize {
        0
    }

    fn write(&mut self, _src: &[T]) -> Result<(), CudaError> {
        Err(CudaError::Unimplemented)
    }

    fn read(&self, _dst: &mut [T]) -> Result<(), CudaError> {
        Err(CudaError::Unimplemented)
    }
}

/// CUDA C2C plan (stub).
pub struct CudaC2cPlan<T: Complex> {
    _marker: PhantomData<T>,
}

impl<T: Complex> C2cPlanOps<CudaBackend, T> for CudaC2cPlan<T> {
    fn execute(
        &mut self,
        _buffer: &mut CudaBuffer<T>,
        _direction: Direction,
    ) -> Result<(), CudaError> {
        Err(CudaError::Unimplemented)
    }
}

/// CUDA R2C plan (stub).
pub struct CudaR2cPlan<F: Real> {
    _marker: PhantomData<F>,
}

impl<F: Real> R2cPlanOps<CudaBackend, F> for CudaR2cPlan<F> {
    fn execute(
        &mut self,
        _input: &CudaBuffer<F>,
        _output: &mut CudaBuffer<F::Complex>,
    ) -> Result<(), CudaError> {
        Err(CudaError::Unimplemented)
    }
}

/// CUDA C2R plan (stub).
pub struct CudaC2rPlan<F: Real> {
    _marker: PhantomData<F>,
}

impl<F: Real> C2rPlanOps<CudaBackend, F> for CudaC2rPlan<F> {
    fn execute(
        &mut self,
        _input: &CudaBuffer<F::Complex>,
        _output: &mut CudaBuffer<F>,
    ) -> Result<(), CudaError> {
        Err(CudaError::Unimplemented)
    }
}
