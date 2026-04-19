//! Backend trait surface.
//!
//! Each GPU backend implements [`Backend`], which ties together a
//! [`Device`], a per-scalar [`Buffer`](Backend::Buffer), and a per-scalar
//! [`Plan`](Backend::Plan). User code can be written either against a
//! specific backend's concrete types (e.g. `vulkan::VulkanBackend`) or
//! generically over any `B: Backend`.

use crate::plan::{Direction, PlanDesc};
use crate::scalar::Scalar;

/// A GPU backend implementation.
///
/// Implementors provide associated types for the device handle, for buffers
/// parameterised by element type, and for plans parameterised by scalar.
/// Error handling is backend-specific; consumers who want cross-backend
/// error handling can constrain `B::Error: Into<MyError>`.
pub trait Backend: Sized + Send + Sync + 'static {
    /// Device handle owning the GPU resources.
    type Device: Device<Self>;
    /// Typed GPU buffer.
    type Buffer<T: Scalar>: BufferOps<Self, T>;
    /// Compiled FFT plan for scalar type `T`.
    type Plan<T: Scalar>: PlanOps<Self, T>;
    /// Error type returned by all fallible backend operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Human-readable backend name (`"vulkan"`, `"cuda"`, ...).
    const NAME: &'static str;
}

/// Operations on a backend's GPU device.
///
/// Construction of the device itself is backend-specific and is therefore
/// not part of this trait; see each backend's `new_device` entry point.
pub trait Device<B: Backend>: Sized + Send + Sync {
    /// Allocate an uninitialised GPU buffer of `len` elements.
    fn alloc<T: Scalar>(&self, len: usize) -> Result<B::Buffer<T>, B::Error>;

    /// Build an FFT plan for the given specification.
    fn plan<T: Scalar>(&self, desc: &PlanDesc) -> Result<B::Plan<T>, B::Error>;

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

/// Operations on a backend FFT plan.
pub trait PlanOps<B: Backend, T: Scalar>: Sized {
    /// Execute the plan in-place on `buffer`, overwriting it with its
    /// transform. The length of `buffer` must equal the plan's expected
    /// total element count (`shape.elements() * batch`).
    fn execute(&mut self, buffer: &mut B::Buffer<T>, direction: Direction) -> Result<(), B::Error>;
}
