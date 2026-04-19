//! Unified GPU-accelerated FFT for Rust.
//!
//! `gpufft` exposes a single trait surface that works the same across
//! multiple GPU backends. Backends are selected at build time via Cargo
//! features; at least one must be enabled.
//!
//! | Feature  | Backend module     | Underlying library |
//! |----------|--------------------|--------------------|
//! | `vulkan` | [`vulkan`]         | VkFFT (vendored)   |
//! | `cuda`   | [`cuda`]           | cuFFT (system)     |
//!
//! # Quick start
//!
//! ```no_run
//! # #[cfg(feature = "vulkan")]
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use gpufft::{
//!     vulkan::VulkanBackend, BufferOps, C2cPlanOps, Device, Direction, PlanDesc, Shape,
//! };
//! use num_complex::Complex32;
//!
//! let device = VulkanBackend::new_device(Default::default())?;
//! let mut buffer = device.alloc::<Complex32>(1024)?;
//! buffer.write(&vec![Complex32::default(); 1024])?;
//!
//! let mut plan = device.plan_c2c::<Complex32>(&PlanDesc {
//!     shape: Shape::D1(1024),
//!     batch: 1,
//!     normalize: false,
//! })?;
//! plan.execute(&mut buffer, Direction::Forward)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Transform kinds
//!
//! Each backend exposes three plan types, constructed through distinct
//! device methods:
//!
//! | Method                           | Trait                           | Use                               |
//! |----------------------------------|---------------------------------|-----------------------------------|
//! | [`Device::plan_c2c`]             | [`C2cPlanOps`]                  | complex-to-complex, in-place     |
//! | [`Device::plan_r2c`]             | [`R2cPlanOps`]                  | real-to-complex, forward only    |
//! | [`Device::plan_c2r`]             | [`C2rPlanOps`]                  | complex-to-real, inverse only    |
//!
//! R2C output buffers and C2R input buffers are Hermitian-symmetric
//! half-spectra (last dimension `n / 2 + 1`), matching VkFFT and cuFFT
//! conventions. Use [`Shape::complex_half_elements`] to size them.
//!
//! Buffers and plans are typed by their backend, so feeding a Vulkan buffer
//! into a CUDA plan is a compile error.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod backend;
pub mod error;
pub mod plan;
pub mod scalar;

#[cfg(feature = "vulkan")]
#[cfg_attr(docsrs, doc(cfg(feature = "vulkan")))]
pub mod vulkan;

#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub mod cuda;

pub use backend::{Backend, BufferOps, C2cPlanOps, C2rPlanOps, Device, R2cPlanOps};
pub use error::Error;
pub use plan::{Direction, PlanDesc, Shape};
pub use scalar::{Complex, Precision, Real, Scalar};
