//! Raw Vulkan handle storage passed to VkFFT.
//!
//! VkFFT's `VkFFTConfiguration` expects pointers to `u64`-width handle values
//! (`VkPhysicalDevice*`, `VkDevice*`, `VkQueue*`, etc.) and may retain those
//! pointers for the lifetime of the application. A [`HandleStorage`] owns
//! one copy of each required handle at a stable address so the pointers
//! remain valid for as long as the backing struct lives.
//!
//! This module is `pub(crate)` plumbing.

use ash::vk::Handle;

/// Stable storage for Vulkan handles referenced by a VkFFT configuration.
///
/// Plans box this struct so its field addresses do not move.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HandleStorage {
    pub(crate) physical_device: u64,
    pub(crate) device: u64,
    pub(crate) queue: u64,
    pub(crate) command_pool: u64,
    pub(crate) fence: u64,
    pub(crate) buffer: u64,
    pub(crate) buffer_size: u64,
}

impl HandleStorage {
    pub(crate) fn new(
        physical_device: ash::vk::PhysicalDevice,
        device: ash::vk::Device,
        queue: ash::vk::Queue,
        command_pool: ash::vk::CommandPool,
        fence: ash::vk::Fence,
        buffer: ash::vk::Buffer,
        buffer_size: u64,
    ) -> Self {
        Self {
            physical_device: physical_device.as_raw(),
            device: device.as_raw(),
            queue: queue.as_raw(),
            command_pool: command_pool.as_raw(),
            fence: fence.as_raw(),
            buffer: buffer.as_raw(),
            buffer_size,
        }
    }
}
