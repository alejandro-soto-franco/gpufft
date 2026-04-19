//! Element types that can flow through an FFT buffer.
//!
//! Three tiers of trait:
//!
//! - [`Scalar`] — any plain-old-data type that survives GPU transfer (`f32`,
//!   `f64`, [`num_complex::Complex32`], [`num_complex::Complex64`]).
//! - [`Real`] — floating-point reals used for R2C / C2R transforms.
//! - [`Complex`] — complex numbers used for C2C / R2C / C2R transforms.
//!
//! [`Real`] and [`Complex`] are paired through their associated types so
//! the trait surface can pin down the partner scalar of an R2C plan:
//! `plan_r2c::<f32>` produces output buffers of `<f32 as Real>::Complex`
//! i.e. [`num_complex::Complex32`].

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

/// Any element type valid inside a [`Buffer`](crate::BufferOps).
pub trait Scalar: Pod + Send + Sync + 'static {
    /// Size of one element in bytes.
    const BYTES: usize = core::mem::size_of::<Self>();
    /// Whether this type is a complex number (vs a bare real).
    const IS_COMPLEX: bool;
    /// Precision class of the underlying real component.
    const PRECISION: Precision;
}

/// A real scalar paired with its natural complex partner.
pub trait Real: Scalar {
    /// The complex type produced by R2C on this real scalar.
    type Complex: Complex<Real = Self>;
}

/// A complex scalar paired with its natural real partner.
pub trait Complex: Scalar {
    /// The real type consumed by R2C to produce this complex scalar.
    type Real: Real<Complex = Self>;
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

impl Real for f32 {
    type Complex = Complex32;
}

impl Real for f64 {
    type Complex = Complex64;
}

impl Complex for Complex32 {
    type Real = f32;
}

impl Complex for Complex64 {
    type Real = f64;
}
