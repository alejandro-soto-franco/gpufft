//! Plan description types.
//!
//! These types are backend-independent. Backends consume a [`PlanDesc`] and
//! return a concrete C2C, R2C, or C2R plan that records dispatches for the
//! requested shape.

/// Shape of an FFT in the real-space domain.
///
/// For C2C transforms this is also the complex buffer shape. For R2C / C2R
/// the complex side is half-sized on the last dimension (Hermitian-symmetric
/// half-spectrum convention, shared by both VkFFT and cuFFT).
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
    /// Real-space element count per batch (product of all dimensions).
    pub fn elements(&self) -> u64 {
        match self {
            Shape::D1(n) => *n as u64,
            Shape::D2([a, b]) => *a as u64 * *b as u64,
            Shape::D3([a, b, c]) => *a as u64 * *b as u64 * *c as u64,
        }
    }

    /// Complex half-spectrum element count per batch. Last dimension is
    /// replaced by `last / 2 + 1`, matching VkFFT and cuFFT conventions for
    /// real transforms.
    pub fn complex_half_elements(&self) -> u64 {
        match self {
            Shape::D1(n) => (*n as u64 / 2) + 1,
            Shape::D2([a, b]) => *a as u64 * ((*b as u64 / 2) + 1),
            Shape::D3([a, b, c]) => *a as u64 * *b as u64 * ((*c as u64 / 2) + 1),
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

/// Direction of an FFT execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Forward transform (time to frequency).
    Forward,
    /// Inverse transform (frequency to time).
    Inverse,
}

impl Direction {
    /// Returns 0 for [`Direction::Forward`], 1 for [`Direction::Inverse`].
    ///
    /// Matches VkFFT's `inverse` flag and cuFFT's `CUFFT_FORWARD` / `CUFFT_INVERSE`.
    pub fn as_int(self) -> i32 {
        match self {
            Direction::Forward => 0,
            Direction::Inverse => 1,
        }
    }
}

/// Specification of an FFT plan.
///
/// The same descriptor is used for C2C, R2C, and C2R; the transform kind is
/// implied by which `plan_*` method is called on the device. `shape` is
/// always the **real-space** shape — for R2C the complex output buffer is
/// half-sized on the last dimension.
#[derive(Clone, Copy, Debug)]
pub struct PlanDesc {
    /// Real-space shape of the transform.
    pub shape: Shape,
    /// Number of independent transforms to batch. Only meaningful for 1D
    /// shapes; must be 1 for 2D/3D.
    pub batch: u32,
    /// When `true`, inverse transforms are scaled by `1 / shape.elements()`
    /// so that forward followed by inverse recovers the input exactly.
    ///
    /// Affects [`C2cPlanOps::execute`](crate::C2cPlanOps::execute) when called
    /// with [`Direction::Inverse`], and every [`C2rPlanOps::execute`](crate::C2rPlanOps::execute)
    /// call. Forward C2C and R2C transforms are always unnormalised.
    ///
    /// When `false` (default, matching cuFFT and rustfft conventions), the
    /// inverse is unnormalised and the composition scales by `shape.elements()`.
    pub normalize: bool,
}

impl Default for PlanDesc {
    fn default() -> Self {
        Self {
            shape: Shape::D1(1),
            batch: 1,
            normalize: false,
        }
    }
}
