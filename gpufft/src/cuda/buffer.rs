//! CUDA-backed FFT buffer: device-local memory allocated via `cudaMalloc`.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

use gpufft_cuda_sys as sys;

use super::device::CudaContext;
use super::error::{CudaError, check_cuda};
use crate::backend::BufferOps;
use crate::scalar::Scalar;

/// A typed GPU buffer in CUDA device memory.
pub struct CudaBuffer<T: Scalar> {
    pub(crate) ctx: Arc<CudaContext>,
    /// Raw CUDA device pointer.
    pub(crate) d_ptr: *mut c_void,
    pub(crate) len: usize,
    pub(crate) size_bytes: u64,
    _marker: PhantomData<T>,
}

// SAFETY: the raw device pointer is only ever dereferenced from the CUDA
// runtime via the owning context, never from Rust code. The context is
// `Send + Sync` through the Arc, so handing the buffer across threads is
// equivalent to handing over the Arc.
unsafe impl<T: Scalar> Send for CudaBuffer<T> {}
// SAFETY: see the `Send` impl above; shared access is safe because we
// never read or write through `d_ptr` from Rust.
unsafe impl<T: Scalar> Sync for CudaBuffer<T> {}

impl<T: Scalar> CudaBuffer<T> {
    pub(crate) fn new(ctx: Arc<CudaContext>, len: usize) -> Result<Self, CudaError> {
        ctx.make_current()?;

        let size_bytes = (len * T::BYTES) as u64;
        let mut d_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `cudaMalloc` writes a device pointer through the provided
        // out-parameter. `size_bytes` is a plain integer count.
        unsafe {
            check_cuda(
                "cudaMalloc",
                sys::cudaMalloc(&mut d_ptr, size_bytes as usize),
            )?;
        }

        Ok(Self {
            ctx,
            d_ptr,
            len,
            size_bytes,
            _marker: PhantomData,
        })
    }

    /// Raw CUDA device pointer (`void*`). Valid for the buffer's lifetime.
    pub fn device_ptr(&self) -> *mut c_void {
        self.d_ptr
    }

    /// Byte size of the allocation.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

impl<T: Scalar> BufferOps<super::CudaBackend, T> for CudaBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn write(&mut self, src: &[T]) -> Result<(), CudaError> {
        if src.len() != self.len {
            return Err(CudaError::LengthMismatch {
                expected: self.len,
                got: src.len(),
            });
        }
        self.ctx.make_current()?;
        // SAFETY: destination is our device pointer (size_bytes large);
        // source is the host slice with matching byte length.
        unsafe {
            check_cuda(
                "cudaMemcpy(host-to-device)",
                sys::cudaMemcpy(
                    self.d_ptr,
                    src.as_ptr().cast::<c_void>(),
                    self.size_bytes as usize,
                    sys::cudaMemcpyKind_cudaMemcpyHostToDevice,
                ),
            )?;
        }
        Ok(())
    }

    fn read(&self, dst: &mut [T]) -> Result<(), CudaError> {
        if dst.len() != self.len {
            return Err(CudaError::LengthMismatch {
                expected: self.len,
                got: dst.len(),
            });
        }
        self.ctx.make_current()?;
        // SAFETY: destination is the host slice (size_bytes large); source
        // is our device pointer.
        unsafe {
            check_cuda(
                "cudaMemcpy(device-to-host)",
                sys::cudaMemcpy(
                    dst.as_mut_ptr().cast::<c_void>(),
                    self.d_ptr,
                    self.size_bytes as usize,
                    sys::cudaMemcpyKind_cudaMemcpyDeviceToHost,
                ),
            )?;
        }
        Ok(())
    }
}

impl<T: Scalar> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `d_ptr` was allocated by `cudaMalloc` and is not yet freed.
        // The ctx Arc ensures `make_current` can still bind the device.
        // Errors on free are swallowed: nothing to do with them on drop.
        unsafe {
            let _ = self.ctx.make_current();
            sys::cudaFree(self.d_ptr);
        }
    }
}
