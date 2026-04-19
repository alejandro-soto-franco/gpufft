//! Vulkan backend (VkFFT).
//!
//! The backend is a pure-[`ash`] Vulkan compute client. No graphics surface
//! is created. Validation layers can be enabled through
//! [`DeviceOptions::enable_validation`].
//!
//! R2C and C2R are scaffolded with typed plan stubs but the implementations
//! are not yet wired; callers get [`VulkanError::Unsupported`]. Use the
//! CUDA backend for R2C / C2R until VkFFT's strided real-buffer layout is
//! integrated.

use crate::backend::Backend;
use crate::scalar::{Complex, Real, Scalar};

mod buffer;
mod device;
mod error;
mod handles;
mod plan;

pub use buffer::VulkanBuffer;
pub use device::{DeviceOptions, VulkanDevice};
pub use error::VulkanError;
pub use plan::{VulkanC2cPlan, VulkanC2rPlan, VulkanR2cPlan};

/// Marker type implementing [`Backend`] for the VkFFT-backed Vulkan backend.
#[derive(Clone, Copy, Debug)]
pub struct VulkanBackend;

impl VulkanBackend {
    /// Construct a new [`VulkanDevice`] using the given options.
    pub fn new_device(options: DeviceOptions) -> Result<VulkanDevice, VulkanError> {
        VulkanDevice::new(options)
    }
}

impl Backend for VulkanBackend {
    type Device = VulkanDevice;
    type Buffer<T: Scalar> = VulkanBuffer<T>;
    type C2cPlan<T: Complex> = VulkanC2cPlan<T>;
    type R2cPlan<F: Real> = VulkanR2cPlan<F>;
    type C2rPlan<F: Real> = VulkanC2rPlan<F>;
    type Error = VulkanError;

    const NAME: &'static str = "vulkan";
}
