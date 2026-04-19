//! CUDA backend (cuFFT). **Stub: all operations return an unimplemented error.**
//!
//! The CUDA backend is scheduled for a subsequent release. This module
//! exists so that `--features cuda` compiles and so backend-generic user
//! code can already be written. Every constructor and every fallible
//! method returns [`CudaError::Unimplemented`].

use std::marker::PhantomData;

use thiserror::Error;

use crate::backend::{Backend, BufferOps, Device, PlanOps};
use crate::plan::{Direction, PlanDesc};
use crate::scalar::Scalar;

/// Errors returned by the CUDA backend stub.
#[derive(Debug, Error)]
pub enum CudaError {
    /// The CUDA backend is not yet implemented.
    #[error("CUDA backend is not yet implemented (gpufft v0.1 stub)")]
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
    type Plan<T: Scalar> = CudaPlan<T>;
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

    fn plan<T: Scalar>(&self, _desc: &PlanDesc) -> Result<CudaPlan<T>, CudaError> {
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

/// CUDA plan (stub).
pub struct CudaPlan<T: Scalar> {
    _marker: PhantomData<T>,
}

impl<T: Scalar> PlanOps<CudaBackend, T> for CudaPlan<T> {
    fn execute(
        &mut self,
        _buffer: &mut CudaBuffer<T>,
        _direction: Direction,
    ) -> Result<(), CudaError> {
        Err(CudaError::Unimplemented)
    }
}
