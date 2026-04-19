//! Vulkan backend (VkFFT).
//!
//! The backend is a pure-[`ash`] Vulkan compute client. No graphics surface
//! is created. Validation layers can be enabled through
//! [`DeviceOptions::enable_validation`].

use crate::backend::Backend;
use crate::scalar::Scalar;

mod buffer;
mod device;
mod error;
mod handles;
mod plan;

pub use buffer::VulkanBuffer;
pub use device::{DeviceOptions, VulkanDevice};
pub use error::VulkanError;
pub use plan::VulkanPlan;

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
    type Plan<T: Scalar> = VulkanPlan<T>;
    type Error = VulkanError;

    const NAME: &'static str = "vulkan";
}
