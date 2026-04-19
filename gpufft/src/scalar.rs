//! Element types that can flow through an FFT buffer.

use bytemuck::Pod;
use num_complex::{Complex32, Complex64};

/// Floating-point precision class of an FFT scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Single precision (IEEE-754 binary32).
    F32,
    /// Double precision (IEEE-754 binary64).
    F64,
}

/// Element types that can flow through an FFT buffer.
///
/// Implemented for `f32`, `f64`, `Complex32`, and `Complex64`. Real types
/// are meaningful only for R2C / C2R transforms, which are planned for a
/// future release; v0.1 consumes complex types only.
pub trait Scalar: Pod + Send + Sync + 'static {
    /// Size of one element in bytes.
    const BYTES: usize = core::mem::size_of::<Self>();
    /// Whether this type is a complex number (vs a bare real).
    const IS_COMPLEX: bool;
    /// Precision class of the underlying real component.
    const PRECISION: Precision;
}

impl Scalar for f32 {
    const IS_COMPLEX: bool = false;
    const PRECISION: Precision = Precision::F32;
}

impl Scalar for f64 {
    const IS_COMPLEX: bool = false;
    const PRECISION: Precision = Precision::F64;
}

impl Scalar for Complex32 {
    const IS_COMPLEX: bool = true;
    const PRECISION: Precision = Precision::F32;
}

impl Scalar for Complex64 {
    const IS_COMPLEX: bool = true;
    const PRECISION: Precision = Precision::F64;
}
