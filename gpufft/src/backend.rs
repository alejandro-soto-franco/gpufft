//! Backend trait surface.
//!
//! Each GPU backend implements [`Backend`], which ties together a
//! [`Device`], a per-scalar [`Buffer`](Backend::Buffer), and three per-scalar
//! plan types: [`C2cPlan`](Backend::C2cPlan) for complex-to-complex in-place
//! transforms, [`R2cPlan`](Backend::R2cPlan) for real-to-complex forward
//! transforms, and [`C2rPlan`](Backend::C2rPlan) for complex-to-real inverse
//! transforms.

use crate::plan::{Direction, PlanDesc};
use crate::scalar::{Complex, Real, Scalar};

/// A GPU backend implementation.
pub trait Backend: Sized + Send + Sync + 'static {
    /// Device handle owning the GPU resources.
    type Device: Device<Self>;
    /// Typed GPU buffer.
    type Buffer<T: Scalar>: BufferOps<Self, T>;
    /// In-place C2C (complex-to-complex) plan.
    type C2cPlan<T: Complex>: C2cPlanOps<Self, T>;
    /// Out-of-place R2C (real-to-complex, forward) plan.
    type R2cPlan<F: Real>: R2cPlanOps<Self, F>;
    /// Out-of-place C2R (complex-to-real, inverse) plan.
    type C2rPlan<F: Real>: C2rPlanOps<Self, F>;
    /// Error type returned by all fallible backend operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Human-readable backend name (`"vulkan"`, `"cuda"`, ...).
    const NAME: &'static str;
}

/// Operations on a backend's GPU device.
pub trait Device<B: Backend>: Sized + Send + Sync {
    /// Allocate an uninitialised GPU buffer of `len` elements.
    fn alloc<T: Scalar>(&self, len: usize) -> Result<B::Buffer<T>, B::Error>;

    /// Build a complex-to-complex in-place plan.
    fn plan_c2c<T: Complex>(&self, desc: &PlanDesc) -> Result<B::C2cPlan<T>, B::Error>;

    /// Build a real-to-complex (forward) plan.
    fn plan_r2c<F: Real>(&self, desc: &PlanDesc) -> Result<B::R2cPlan<F>, B::Error>;

    /// Build a complex-to-real (inverse) plan.
    fn plan_c2r<F: Real>(&self, desc: &PlanDesc) -> Result<B::C2rPlan<F>, B::Error>;

    /// Block until all in-flight GPU work on this device completes.
    fn synchronize(&self) -> Result<(), B::Error>;
}

/// Operations on a backend buffer.
pub trait BufferOps<B: Backend, T: Scalar>: Sized {
    /// Number of elements.
    fn len(&self) -> usize;

    /// `true` if `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upload `src` into the buffer. `src.len()` must equal `self.len()`.
    fn write(&mut self, src: &[T]) -> Result<(), B::Error>;

    /// Download the buffer into `dst`. `dst.len()` must equal `self.len()`.
    fn read(&self, dst: &mut [T]) -> Result<(), B::Error>;
}

/// Operations on an in-place C2C plan.
pub trait C2cPlanOps<B: Backend, T: Complex>: Sized {
    /// Execute the plan in-place on `buffer`, overwriting it with its
    /// transform. `buffer.len()` must equal `shape.elements() * batch`.
    fn execute(&mut self, buffer: &mut B::Buffer<T>, direction: Direction) -> Result<(), B::Error>;
}

/// Operations on an R2C (real-to-complex forward) plan.
pub trait R2cPlanOps<B: Backend, F: Real>: Sized {
    /// Execute the R2C transform. `input.len()` must equal
    /// `shape.elements() * batch`; `output.len()` must equal
    /// `shape.complex_half_elements() * batch`.
    fn execute(
        &mut self,
        input: &B::Buffer<F>,
        output: &mut B::Buffer<F::Complex>,
    ) -> Result<(), B::Error>;
}

/// Operations on a C2R (complex-to-real inverse) plan.
pub trait C2rPlanOps<B: Backend, F: Real>: Sized {
    /// Execute the C2R transform. `input.len()` must equal
    /// `shape.complex_half_elements() * batch`; `output.len()` must equal
    /// `shape.elements() * batch`.
    fn execute(
        &mut self,
        input: &B::Buffer<F::Complex>,
        output: &mut B::Buffer<F>,
    ) -> Result<(), B::Error>;
}
