//! Plan description types.
//!
//! These types are backend-independent. Backends consume a [`PlanDesc`] and
//! return a concrete plan that records dispatches for the requested shape
//! and transform.

/// Shape of an FFT.
///
/// VkFFT and cuFFT both support 1D, 2D, and 3D transforms natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// One-dimensional FFT of length `n`.
    D1(u32),
    /// Two-dimensional FFT of extents `[nx, ny]`.
    D2([u32; 2]),
    /// Three-dimensional FFT of extents `[nx, ny, nz]`.
    D3([u32; 3]),
}

impl Shape {
    /// Total number of elements per batch.
    pub fn elements(&self) -> u64 {
        match self {
            Shape::D1(n) => *n as u64,
            Shape::D2([a, b]) => *a as u64 * *b as u64,
            Shape::D3([a, b, c]) => *a as u64 * *b as u64 * *c as u64,
        }
    }

    /// Dimensionality of the transform (1, 2, or 3).
    pub fn rank(&self) -> u8 {
        match self {
            Shape::D1(_) => 1,
            Shape::D2(_) => 2,
            Shape::D3(_) => 3,
        }
    }
}

/// Transform kind.
///
/// Only [`Transform::C2c`] is implemented in v0.1. The other variants are
/// reserved so the API does not need to change when R2C/C2R support lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// Complex-to-complex.
    C2c,
    /// Real-to-complex (half-spectrum output per VkFFT/cuFFT convention).
    /// *Not implemented in v0.1.*
    R2c,
    /// Complex-to-real (inverse of [`Transform::R2c`]).
    /// *Not implemented in v0.1.*
    C2r,
}

/// Direction of an FFT execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Forward transform (sign convention follows the backend).
    Forward,
    /// Inverse transform.
    Inverse,
}

impl Direction {
    /// Returns 0 for [`Direction::Forward`], 1 for [`Direction::Inverse`].
    ///
    /// Matches both VkFFT's `inverse` flag and cuFFT's `CUFFT_FORWARD`/
    /// `CUFFT_INVERSE` integer constants.
    pub fn as_int(self) -> i32 {
        match self {
            Direction::Forward => 0,
            Direction::Inverse => 1,
        }
    }
}

/// Specification of an FFT plan.
#[derive(Clone, Copy, Debug)]
pub struct PlanDesc {
    /// Shape of the transform.
    pub shape: Shape,
    /// Kind of transform.
    pub transform: Transform,
    /// Number of independent transforms to batch. Only meaningful for
    /// 1D shapes in v0.1; must be 1 for 2D/3D.
    pub batch: u32,
    /// When `true`, the inverse transform is scaled by `1 / shape.elements()`
    /// so that `forward` followed by `inverse` recovers the input exactly.
    /// When `false` (the default, matching cuFFT and rustfft conventions),
    /// the inverse is unnormalised and the composition scales by
    /// `shape.elements()`.
    pub normalize: bool,
}

impl Default for PlanDesc {
    fn default() -> Self {
        Self {
            shape: Shape::D1(1),
            transform: Transform::C2c,
            batch: 1,
            normalize: false,
        }
    }
}
