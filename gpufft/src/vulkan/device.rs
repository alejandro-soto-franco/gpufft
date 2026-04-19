//! Vulkan device construction and shared context.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use ash::vk;

use super::buffer::VulkanBuffer;
use super::error::VulkanError;
use super::plan::{VulkanC2cPlan, VulkanC2rPlan, VulkanR2cPlan};
use crate::backend::Device;
use crate::plan::PlanDesc;
use crate::scalar::{Complex, Real, Scalar};

/// Options controlling [`VulkanDevice`] construction.
#[derive(Clone, Debug, Default)]
pub struct DeviceOptions {
    /// Index into the list of Vulkan physical devices. If `None`, selects
    /// the first discrete GPU, falling back to the first device of any kind.
    pub preferred_device_index: Option<usize>,
    /// Enable `VK_LAYER_KHRONOS_validation` at instance creation. Requires
    /// the validation layer to be installed on the system.
    pub enable_validation: bool,
}

/// Shared Vulkan state held by [`VulkanDevice`], [`VulkanBuffer`], and
/// [`VulkanPlan`] via [`Arc`] so resources can be cleaned up in any drop order.
pub(crate) struct VulkanContext {
    // Held to keep the dynamically-loaded Vulkan loader alive for the
    // lifetime of the Instance, Device, and every resource that depends on
    // it. Never read after construction.
    #[allow(dead_code)]
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) queue: vk::Queue,
    pub(crate) queue_family_index: u32,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) transfer_pool: vk::CommandPool,
    pub(crate) transfer_fence: vk::Fence,
}

impl VulkanContext {
    pub(crate) fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, VulkanError> {
        for i in 0..self.memory_properties.memory_type_count {
            let mt = &self.memory_properties.memory_types[i as usize];
            if (type_filter & (1 << i)) != 0 && mt.property_flags.contains(properties) {
                return Ok(i);
            }
        }
        Err(VulkanError::NoSuitableMemoryType)
    }

    /// Allocate, bind, and return a buffer + memory pair.
    pub(crate) fn allocate_buffer(
        &self,
        size_bytes: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, u64), VulkanError> {
        // SAFETY: device and instance are valid for the lifetime of `self`.
        let buffer = unsafe {
            let ci = vk::BufferCreateInfo::default()
                .size(size_bytes)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            self.device
                .create_buffer(&ci, None)
                .map_err(|e| VulkanError::vk("create_buffer", e))?
        };

        // SAFETY: buffer was just created by us.
        let mem_req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_type = self.find_memory_type(mem_req.memory_type_bits, properties)?;

        // SAFETY: same.
        let memory = unsafe {
            let ai = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_type);
            self.device.allocate_memory(&ai, None).map_err(|e| {
                // Ensure the buffer is destroyed if allocation fails.
                self.device.destroy_buffer(buffer, None);
                VulkanError::vk("allocate_memory", e)
            })?
        };

        // SAFETY: both handles are freshly created and owned by us.
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                    VulkanError::vk("bind_buffer_memory", e)
                })?;
        }

        Ok((buffer, memory, mem_req.size))
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: device outlives the resources it owns; teardown order is
        // fence before pool before device before instance.
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_fence(self.transfer_fence, None);
            self.device.destroy_command_pool(self.transfer_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A Vulkan compute device, bound to a single physical device and a single
/// compute-capable queue.
pub struct VulkanDevice {
    pub(crate) ctx: Arc<VulkanContext>,
}

impl VulkanDevice {
    /// Construct a new device from the given options.
    pub fn new(options: DeviceOptions) -> Result<Self, VulkanError> {
        // SAFETY: `ash::Entry::load` loads the system Vulkan loader. No
        // invariants beyond the platform providing libvulkan.
        let entry = unsafe { ash::Entry::load().map_err(VulkanError::LoaderLoad)? };

        let app_name = CString::new("gpufft").unwrap();
        let engine_name = CString::new("gpufft").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let mut enabled_layers: Vec<*const i8> = Vec::new();
        if options.enable_validation {
            // SAFETY: enumerate_instance_layer_properties is a safe Vulkan entry.
            let available_layers = unsafe {
                entry
                    .enumerate_instance_layer_properties()
                    .map_err(|e| VulkanError::vk("enumerate_instance_layer_properties", e))?
            };
            let found = available_layers.iter().any(|l| {
                // SAFETY: layer_name is a fixed-size NUL-terminated buffer.
                let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
                name == validation_layer.as_c_str()
            });
            if !found {
                return Err(VulkanError::ValidationUnavailable(
                    "VK_LAYER_KHRONOS_validation".into(),
                ));
            }
            enabled_layers.push(validation_layer.as_ptr());
        }

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&enabled_layers);

        // SAFETY: instance creation parameters are valid for the call duration.
        let instance = unsafe {
            entry
                .create_instance(&instance_ci, None)
                .map_err(|e| VulkanError::vk("create_instance", e))?
        };

        // SAFETY: instance just created.
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|e| VulkanError::vk("enumerate_physical_devices", e))?
        };

        let pd_index = match options.preferred_device_index {
            Some(i) => i,
            None => pick_discrete_or_first(&instance, &physical_devices),
        };
        let physical_device = *physical_devices
            .get(pd_index)
            .ok_or(VulkanError::NoDevice)?;

        // SAFETY: physical device is valid.
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_family_properties
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or(VulkanError::NoDevice)? as u32;

        let queue_priorities = [1.0f32];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];

        let device_ci = vk::DeviceCreateInfo::default().queue_create_infos(&queue_ci);

        // SAFETY: instance, physical_device, and queue_ci are all valid.
        let device = unsafe {
            instance
                .create_device(physical_device, &device_ci, None)
                .map_err(|e| VulkanError::vk("create_device", e))?
        };

        // SAFETY: device just created; queue index is within queue_ci[0].
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // SAFETY: instance and physical_device are valid.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // SAFETY: device just created; queue_family_index was validated above.
        let transfer_pool = unsafe {
            let ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            device
                .create_command_pool(&ci, None)
                .map_err(|e| VulkanError::vk("create_command_pool", e))?
        };

        // SAFETY: device just created.
        let transfer_fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| VulkanError::vk("create_fence", e))?
        };

        let ctx = Arc::new(VulkanContext {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            memory_properties,
            transfer_pool,
            transfer_fence,
        });

        Ok(Self { ctx })
    }

    /// Return the name of the selected physical device.
    pub fn adapter_name(&self) -> String {
        // SAFETY: instance and physical_device are valid.
        let props = unsafe {
            self.ctx
                .instance
                .get_physical_device_properties(self.ctx.physical_device)
        };
        // SAFETY: device_name is a fixed-size NUL-terminated buffer.
        let cstr = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }
}

impl Device<super::VulkanBackend> for VulkanDevice {
    fn alloc<T: Scalar>(&self, len: usize) -> Result<VulkanBuffer<T>, VulkanError> {
        VulkanBuffer::new(self.ctx.clone(), len)
    }

    fn plan_c2c<T: Complex>(&self, desc: &PlanDesc) -> Result<VulkanC2cPlan<T>, VulkanError> {
        VulkanC2cPlan::new(self.ctx.clone(), *desc)
    }

    fn plan_r2c<F: Real>(&self, desc: &PlanDesc) -> Result<VulkanR2cPlan<F>, VulkanError> {
        VulkanR2cPlan::new(self.ctx.clone(), *desc)
    }

    fn plan_c2r<F: Real>(&self, desc: &PlanDesc) -> Result<VulkanC2rPlan<F>, VulkanError> {
        VulkanC2rPlan::new(self.ctx.clone(), *desc)
    }

    fn synchronize(&self) -> Result<(), VulkanError> {
        // SAFETY: device is valid.
        unsafe {
            self.ctx
                .device
                .device_wait_idle()
                .map_err(|e| VulkanError::vk("device_wait_idle", e))
        }
    }
}

fn pick_discrete_or_first(instance: &ash::Instance, devices: &[vk::PhysicalDevice]) -> usize {
    for (i, &pd) in devices.iter().enumerate() {
        // SAFETY: pd is from enumerate_physical_devices on `instance`.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            return i;
        }
    }
    0
}
