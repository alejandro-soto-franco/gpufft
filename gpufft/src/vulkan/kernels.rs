//! Compute-shader kernels for the Vulkan backend.
//!
//! Currently provides [`StrideCopyKernel`], a single compute pipeline
//! that performs stride-aware buffer copies. Used by R2C / C2R plans to
//! pad / strip innermost rows without issuing one `vkCmdCopyBuffer`
//! region per row.

use std::sync::Arc;

use ash::vk;

use super::device::VulkanContext;
use super::error::VulkanError;

// Reduce the read_spv error path below to our existing VkResult variant.
impl VulkanError {
    fn with_context(self, _io: std::io::Error) -> Self {
        self
    }
}

/// Compiled SPIR-V for `shaders/stride_copy.comp`, embedded at build time.
const STRIDE_COPY_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stride_copy.spv"));

/// Local workgroup size declared in `stride_copy.comp`.
const STRIDE_COPY_LOCAL_SIZE_X: u32 = 64;

/// Compute-shader kernel performing stride-aware buffer copies.
///
/// Descriptor layout: two storage buffers (binding 0 = src, binding 1 =
/// dst). Push constants: four u32s (`row_uints`, `src_stride_uints`,
/// `dst_stride_uints`, `n_rows`). All sizes and strides are in uint
/// (4-byte) units so a single pipeline handles every scalar precision.
pub(crate) struct StrideCopyKernel {
    ctx: Arc<VulkanContext>,
    shader_module: vk::ShaderModule,
    dsl: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

/// Push-constant block matching `stride_copy.comp`'s `Push` uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PushBlock {
    row_uints: u32,
    src_stride_uints: u32,
    dst_stride_uints: u32,
    n_rows: u32,
}

// SAFETY: POD. All fields are u32 with no padding.
unsafe impl bytemuck::Zeroable for PushBlock {}
// SAFETY: same; bytemuck requires the type to be Plain-Old-Data.
unsafe impl bytemuck::Pod for PushBlock {}

impl StrideCopyKernel {
    pub(crate) fn new(ctx: Arc<VulkanContext>) -> Result<Self, VulkanError> {
        // `include_bytes!` returns an unaligned `&[u8]`; copy into a u32-
        // aligned Vec via ash's helper before handing to Vulkan.
        let code_u32 =
            ash::util::read_spv(&mut std::io::Cursor::new(STRIDE_COPY_SPV)).map_err(|e| {
                VulkanError::vk("read_spv", ash::vk::Result::ERROR_INITIALIZATION_FAILED)
                    .with_context(e)
            })?;

        // SAFETY: code_u32 is a valid SPIR-V binary produced by our build.rs.
        let shader_module = unsafe {
            let ci = vk::ShaderModuleCreateInfo::default().code(&code_u32);
            ctx.device
                .create_shader_module(&ci, None)
                .map_err(|e| VulkanError::vk("create_shader_module", e))?
        };

        // SAFETY: ctx.device is valid.
        let dsl = unsafe {
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            ctx.device
                .create_descriptor_set_layout(&ci, None)
                .map_err(|e| {
                    ctx.device.destroy_shader_module(shader_module, None);
                    VulkanError::vk("create_descriptor_set_layout", e)
                })?
        };

        // SAFETY: dsl just created.
        let pipeline_layout = unsafe {
            let push_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<PushBlock>() as u32)];
            let dsls = [dsl];
            let ci = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&dsls)
                .push_constant_ranges(&push_range);
            ctx.device.create_pipeline_layout(&ci, None).map_err(|e| {
                ctx.device.destroy_descriptor_set_layout(dsl, None);
                ctx.device.destroy_shader_module(shader_module, None);
                VulkanError::vk("create_pipeline_layout", e)
            })?
        };

        // SAFETY: shader_module and pipeline_layout just created.
        let pipeline = unsafe {
            let entry_name = std::ffi::CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_name);
            let ci = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)];
            let pipelines = ctx
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &ci, None)
                .map_err(|(_, e)| {
                    ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                    ctx.device.destroy_descriptor_set_layout(dsl, None);
                    ctx.device.destroy_shader_module(shader_module, None);
                    VulkanError::vk("create_compute_pipelines", e)
                })?;
            pipelines[0]
        };

        // SAFETY: ctx.device is valid.
        let descriptor_pool = unsafe {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2)];
            let ci = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes);
            ctx.device.create_descriptor_pool(&ci, None).map_err(|e| {
                ctx.device.destroy_pipeline(pipeline, None);
                ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                ctx.device.destroy_descriptor_set_layout(dsl, None);
                ctx.device.destroy_shader_module(shader_module, None);
                VulkanError::vk("create_descriptor_pool", e)
            })?
        };

        // SAFETY: pool + dsl valid.
        let descriptor_set = unsafe {
            let dsls = [dsl];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&dsls);
            ctx.device.allocate_descriptor_sets(&alloc).map_err(|e| {
                ctx.device.destroy_descriptor_pool(descriptor_pool, None);
                ctx.device.destroy_pipeline(pipeline, None);
                ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                ctx.device.destroy_descriptor_set_layout(dsl, None);
                ctx.device.destroy_shader_module(shader_module, None);
                VulkanError::vk("allocate_descriptor_sets", e)
            })?[0]
        };

        Ok(Self {
            ctx,
            shader_module,
            dsl,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_set,
        })
    }

    /// Update the descriptor set to point at `src` and `dst`. Must be
    /// called outside any command-buffer recording and before the next
    /// [`Self::record_dispatch`]. The previous dispatch using this
    /// descriptor set must have already completed (guaranteed by the
    /// plan's fence wait between executes).
    pub(crate) fn update_descriptor(
        &self,
        src: vk::Buffer,
        src_bytes: u64,
        dst: vk::Buffer,
        dst_bytes: u64,
    ) {
        // SAFETY: ctx.device + descriptor_set valid.
        unsafe {
            let src_info = [vk::DescriptorBufferInfo::default()
                .buffer(src)
                .offset(0)
                .range(src_bytes)];
            let dst_info = [vk::DescriptorBufferInfo::default()
                .buffer(dst)
                .offset(0)
                .range(dst_bytes)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&src_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&dst_info),
            ];
            self.ctx.device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Record a dispatch that copies `n_rows` rows of `row_uints`
    /// uint-words each, using the given source/destination row strides.
    /// Sizes are in uint (4-byte) units.
    pub(crate) fn record_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        row_uints: u32,
        src_stride_uints: u32,
        dst_stride_uints: u32,
        n_rows: u32,
    ) {
        let pc = PushBlock {
            row_uints,
            src_stride_uints,
            dst_stride_uints,
            n_rows,
        };
        let total = row_uints as u64 * n_rows as u64;
        let groups_x = total.div_ceil(STRIDE_COPY_LOCAL_SIZE_X as u64) as u32;

        // SAFETY: cmd is in recording state; pipeline + descriptor set valid.
        unsafe {
            self.ctx
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            self.ctx.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            self.ctx.device.cmd_dispatch(cmd, groups_x, 1, 1);
        }
    }
}

impl Drop for StrideCopyKernel {
    fn drop(&mut self) {
        // SAFETY: all objects created by us and not yet destroyed.
        unsafe {
            self.ctx.device.device_wait_idle().ok();
            self.ctx
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.dsl, None);
            self.ctx
                .device
                .destroy_shader_module(self.shader_module, None);
        }
    }
}
