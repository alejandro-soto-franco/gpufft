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
//!     vulkan::VulkanBackend, BufferOps, Device, Direction, PlanDesc, PlanOps, Shape,
//!     Transform,
//! };
//! use num_complex::Complex32;
//!
//! let device = VulkanBackend::new_device(Default::default())?;
//! let mut buffer = device.alloc::<Complex32>(1024)?;
//! buffer.write(&vec![Complex32::default(); 1024])?;
//!
//! let mut plan = device.plan::<Complex32>(&PlanDesc {
//!     shape: Shape::D1(1024),
//!     transform: Transform::C2c,
//!     batch: 1,
//!     normalize: false,
//! })?;
//! plan.execute(&mut buffer, Direction::Forward)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Scope (v0.1)
//!
//! The initial release supports complex-to-complex, in-place FFTs in 1D, 2D,
//! and 3D, at single and double precision. Real-to-complex (R2C) and
//! complex-to-real (C2R) transforms, out-of-place execution, and DCT/DST
//! variants are planned for later releases.
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

pub use backend::{Backend, BufferOps, Device, PlanOps};
pub use error::Error;
pub use plan::{Direction, PlanDesc, Shape, Transform};
pub use scalar::{Precision, Scalar};
