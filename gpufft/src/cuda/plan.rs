//! CUDA cuFFT plan types for C2C, R2C, and C2R transforms.
//!
//! Precision is dispatched by the scalar type:
//!
//! | Scalar     | C2C type  | R2C type  | C2R type  |
//! |------------|-----------|-----------|-----------|
//! | Complex32  | CUFFT_C2C | CUFFT_R2C | CUFFT_C2R |
//! | Complex64  | CUFFT_Z2Z | CUFFT_D2Z | CUFFT_Z2D |

use std::marker::PhantomData;
use std::sync::Arc;

use gpufft_cuda_sys as sys;

use super::buffer::CudaBuffer;
use super::device::CudaContext;
use super::error::{CudaError, check_cufft};
use crate::backend::{C2cPlanOps, C2rPlanOps, R2cPlanOps};
use crate::plan::{Direction, PlanDesc, Shape};
use crate::scalar::{Complex, Precision, Real};

/// Plan for a complex-to-complex in-place FFT on CUDA.
pub struct CudaC2cPlan<T: Complex> {
    ctx: Arc<CudaContext>,
    plan: sys::cufftHandle,
    element_count: usize,
    _marker: PhantomData<T>,
}

impl<T: Complex> CudaC2cPlan<T> {
    pub(crate) fn new(ctx: Arc<CudaContext>, desc: PlanDesc) -> Result<Self, CudaError> {
        validate_desc(&desc)?;

        let element_count = (desc.shape.elements() * desc.batch as u64) as usize;
        let cufft_type = match T::PRECISION {
            Precision::F32 => sys::cufftType_t_CUFFT_C2C,
            Precision::F64 => sys::cufftType_t_CUFFT_Z2Z,
        };

        ctx.make_current()?;
        let plan = cufft_plan_for_shape(&desc.shape, cufft_type, desc.batch)?;

        Ok(Self {
            ctx,
            plan,
            element_count,
            _marker: PhantomData,
        })
    }
}

impl<T: Complex> C2cPlanOps<super::CudaBackend, T> for CudaC2cPlan<T> {
    fn execute(
        &mut self,
        buffer: &mut CudaBuffer<T>,
        direction: Direction,
    ) -> Result<(), CudaError> {
        if buffer.len != self.element_count {
            return Err(CudaError::LengthMismatch {
                expected: self.element_count,
                got: buffer.len,
            });
        }

        // bindgen emits CUFFT_FORWARD as i32 (=-1) and CUFFT_INVERSE as u32
        // (=1); cufftExecC2C takes a plain `int`, so unify to i32.
        let sign: i32 = match direction {
            Direction::Forward => sys::CUFFT_FORWARD,
            Direction::Inverse => sys::CUFFT_INVERSE as i32,
        };

        self.ctx.make_current()?;
        // SAFETY: `plan` was created by `cufftPlan*`; `buffer.d_ptr` is a
        // device pointer we own with byte size matching the plan extent.
        // For in-place C2C we pass the same pointer for input and output.
        unsafe {
            let ptr = buffer.d_ptr;
            let code = match T::PRECISION {
                Precision::F32 => sys::cufftExecC2C(
                    self.plan,
                    ptr.cast::<sys::cufftComplex>(),
                    ptr.cast::<sys::cufftComplex>(),
                    sign,
                ),
                Precision::F64 => sys::cufftExecZ2Z(
                    self.plan,
                    ptr.cast::<sys::cufftDoubleComplex>(),
                    ptr.cast::<sys::cufftDoubleComplex>(),
                    sign,
                ),
            };
            check_cufft("cufftExec (C2C/Z2Z)", code)?;
            check_cufft(
                "cudaDeviceSynchronize",
                map_cuda_to_cufft(sys::cudaDeviceSynchronize()),
            )?;
        }
        Ok(())
    }
}

impl<T: Complex> Drop for CudaC2cPlan<T> {
    fn drop(&mut self) {
        // SAFETY: `plan` was produced by `cufftPlan*` and is not yet destroyed.
        unsafe {
            let _ = self.ctx.make_current();
            sys::cufftDestroy(self.plan);
        }
    }
}

/// Plan for a real-to-complex forward FFT on CUDA.
pub struct CudaR2cPlan<F: Real> {
    ctx: Arc<CudaContext>,
    plan: sys::cufftHandle,
    real_element_count: usize,
    complex_element_count: usize,
    _marker: PhantomData<F>,
}

impl<F: Real> CudaR2cPlan<F> {
    pub(crate) fn new(ctx: Arc<CudaContext>, desc: PlanDesc) -> Result<Self, CudaError> {
        validate_desc(&desc)?;

        let real_element_count = (desc.shape.elements() * desc.batch as u64) as usize;
        let complex_element_count =
            (desc.shape.complex_half_elements() * desc.batch as u64) as usize;

        let cufft_type = match F::PRECISION {
            Precision::F32 => sys::cufftType_t_CUFFT_R2C,
            Precision::F64 => sys::cufftType_t_CUFFT_D2Z,
        };

        ctx.make_current()?;
        let plan = cufft_plan_for_shape(&desc.shape, cufft_type, desc.batch)?;

        Ok(Self {
            ctx,
            plan,
            real_element_count,
            complex_element_count,
            _marker: PhantomData,
        })
    }
}

impl<F: Real> R2cPlanOps<super::CudaBackend, F> for CudaR2cPlan<F> {
    fn execute(
        &mut self,
        input: &CudaBuffer<F>,
        output: &mut CudaBuffer<F::Complex>,
    ) -> Result<(), CudaError> {
        if input.len != self.real_element_count {
            return Err(CudaError::LengthMismatch {
                expected: self.real_element_count,
                got: input.len,
            });
        }
        if output.len != self.complex_element_count {
            return Err(CudaError::LengthMismatch {
                expected: self.complex_element_count,
                got: output.len,
            });
        }

        self.ctx.make_current()?;
        // SAFETY: both pointers are ours with matching element counts.
        unsafe {
            let code = match F::PRECISION {
                Precision::F32 => sys::cufftExecR2C(
                    self.plan,
                    input.d_ptr.cast::<sys::cufftReal>(),
                    output.d_ptr.cast::<sys::cufftComplex>(),
                ),
                Precision::F64 => sys::cufftExecD2Z(
                    self.plan,
                    input.d_ptr.cast::<sys::cufftDoubleReal>(),
                    output.d_ptr.cast::<sys::cufftDoubleComplex>(),
                ),
            };
            check_cufft("cufftExec (R2C/D2Z)", code)?;
            check_cufft(
                "cudaDeviceSynchronize",
                map_cuda_to_cufft(sys::cudaDeviceSynchronize()),
            )?;
        }
        Ok(())
    }
}

impl<F: Real> Drop for CudaR2cPlan<F> {
    fn drop(&mut self) {
        // SAFETY: `plan` was produced by `cufftPlan*` and is not yet destroyed.
        unsafe {
            let _ = self.ctx.make_current();
            sys::cufftDestroy(self.plan);
        }
    }
}

/// Plan for a complex-to-real inverse FFT on CUDA.
pub struct CudaC2rPlan<F: Real> {
    ctx: Arc<CudaContext>,
    plan: sys::cufftHandle,
    complex_element_count: usize,
    real_element_count: usize,
    _marker: PhantomData<F>,
}

impl<F: Real> CudaC2rPlan<F> {
    pub(crate) fn new(ctx: Arc<CudaContext>, desc: PlanDesc) -> Result<Self, CudaError> {
        validate_desc(&desc)?;

        let real_element_count = (desc.shape.elements() * desc.batch as u64) as usize;
        let complex_element_count =
            (desc.shape.complex_half_elements() * desc.batch as u64) as usize;

        let cufft_type = match F::PRECISION {
            Precision::F32 => sys::cufftType_t_CUFFT_C2R,
            Precision::F64 => sys::cufftType_t_CUFFT_Z2D,
        };

        ctx.make_current()?;
        let plan = cufft_plan_for_shape(&desc.shape, cufft_type, desc.batch)?;

        Ok(Self {
            ctx,
            plan,
            real_element_count,
            complex_element_count,
            _marker: PhantomData,
        })
    }
}

impl<F: Real> C2rPlanOps<super::CudaBackend, F> for CudaC2rPlan<F> {
    fn execute(
        &mut self,
        input: &CudaBuffer<F::Complex>,
        output: &mut CudaBuffer<F>,
    ) -> Result<(), CudaError> {
        if input.len != self.complex_element_count {
            return Err(CudaError::LengthMismatch {
                expected: self.complex_element_count,
                got: input.len,
            });
        }
        if output.len != self.real_element_count {
            return Err(CudaError::LengthMismatch {
                expected: self.real_element_count,
                got: output.len,
            });
        }

        self.ctx.make_current()?;
        // SAFETY: both pointers are ours with matching element counts.
        unsafe {
            let code = match F::PRECISION {
                Precision::F32 => sys::cufftExecC2R(
                    self.plan,
                    input.d_ptr.cast::<sys::cufftComplex>(),
                    output.d_ptr.cast::<sys::cufftReal>(),
                ),
                Precision::F64 => sys::cufftExecZ2D(
                    self.plan,
                    input.d_ptr.cast::<sys::cufftDoubleComplex>(),
                    output.d_ptr.cast::<sys::cufftDoubleReal>(),
                ),
            };
            check_cufft("cufftExec (C2R/Z2D)", code)?;
            check_cufft(
                "cudaDeviceSynchronize",
                map_cuda_to_cufft(sys::cudaDeviceSynchronize()),
            )?;
        }
        Ok(())
    }
}

impl<F: Real> Drop for CudaC2rPlan<F> {
    fn drop(&mut self) {
        // SAFETY: `plan` was produced by `cufftPlan*` and is not yet destroyed.
        unsafe {
            let _ = self.ctx.make_current();
            sys::cufftDestroy(self.plan);
        }
    }
}

// ---------- helpers ----------

fn validate_desc(desc: &PlanDesc) -> Result<(), CudaError> {
    if desc.normalize {
        return Err(CudaError::UnsupportedNormalize);
    }
    if desc.batch == 0 {
        return Err(CudaError::InvalidPlan("batch must be at least 1"));
    }
    if desc.batch > 1 && desc.shape.rank() > 1 {
        return Err(CudaError::InvalidPlan(
            "batch > 1 is supported only for 1D shapes in v0.2",
        ));
    }
    Ok(())
}

fn cufft_plan_for_shape(
    shape: &Shape,
    cufft_type: sys::cufftType,
    batch: u32,
) -> Result<sys::cufftHandle, CudaError> {
    let mut plan: sys::cufftHandle = 0;
    // SAFETY: `plan` is written through by cufftPlan*; all size arguments
    // are plain integers.
    let code = unsafe {
        match *shape {
            Shape::D1(n) => sys::cufftPlan1d(&mut plan, n as i32, cufft_type, batch as i32),
            Shape::D2([nx, ny]) => sys::cufftPlan2d(&mut plan, nx as i32, ny as i32, cufft_type),
            Shape::D3([nx, ny, nz]) => {
                sys::cufftPlan3d(&mut plan, nx as i32, ny as i32, nz as i32, cufft_type)
            }
        }
    };
    check_cufft("cufftPlan*", code)?;
    Ok(plan)
}

/// Map a CUDA Runtime error to a `cufftResult` bucket so the generic
/// `check_cufft` wrapper can consume both call-sites uniformly.
fn map_cuda_to_cufft(code: sys::cudaError_t) -> sys::cufftResult {
    if code == sys::cudaError_cudaSuccess {
        sys::cufftResult_t_CUFFT_SUCCESS
    } else {
        sys::cufftResult_t_CUFFT_EXEC_FAILED
    }
}
