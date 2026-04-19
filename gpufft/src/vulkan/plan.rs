//! Vulkan FFT plan wrapping a VkFFT application.
//!
//! A plan owns:
//!
//! - a dedicated `VkFFTApplication`,
//! - a command pool and fence used exclusively for FFT dispatches,
//! - an internal device-local VkBuffer that VkFFT reads and writes, and
//! - [`HandleStorage`](super::handles::HandleStorage) whose u64 fields back
//!   the pointer slots in `VkFFTConfiguration`.
//!
//! The handle storage is the critical invariant: VkFFT retains pointers
//! from the configuration for later `VkFFTAppend` calls, so the u64
//! addresses must be stable for the plan's lifetime. This is why the plan
//! boxes its inner state.

use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;
use gpufft_vulkan_sys as sys;

use super::buffer::VulkanBuffer;
use super::device::VulkanContext;
use super::error::VulkanError;
use super::handles::HandleStorage;
use crate::backend::PlanOps;
use crate::plan::{Direction, PlanDesc, Shape, Transform};
use crate::scalar::{Precision, Scalar};

/// A VkFFT-backed FFT plan.
pub struct VulkanPlan<T: Scalar> {
    inner: Box<PlanInner>,
    _marker: PhantomData<T>,
}

struct PlanInner {
    ctx: Arc<VulkanContext>,
    /// VkFFT application. Opaque to us; populated by `initializeVkFFT`.
    app: sys::VkFFTApplication,
    /// Stable-address handle storage. `app` holds pointers into this via
    /// `VkFFTConfiguration`. Lives exactly as long as `app`.
    handles: HandleStorage,
    /// Command pool used for this plan's dispatches.
    command_pool: vk::CommandPool,
    /// Fence reused across every `execute` call.
    fence: vk::Fence,
    /// Internal device-local VkFFT buffer.
    fft_buffer: vk::Buffer,
    fft_memory: vk::DeviceMemory,
    /// Element count and byte size of the internal buffer.
    element_count: usize,
    size_bytes: u64,
    desc: PlanDesc,
}

impl<T: Scalar> VulkanPlan<T> {
    pub(crate) fn new(ctx: Arc<VulkanContext>, desc: PlanDesc) -> Result<Self, VulkanError> {
        validate_desc::<T>(&desc)?;

        let element_count = (desc.shape.elements() * desc.batch as u64) as usize;
        let size_bytes = (element_count * T::BYTES) as u64;

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let (fft_buffer, fft_memory, _) = ctx.allocate_buffer(
            size_bytes,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Dedicated pool so concurrent plans do not fight for a shared pool.
        // SAFETY: ctx.device is valid; queue_family_index is the compute queue.
        let command_pool = unsafe {
            let ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            ctx.device.create_command_pool(&ci, None).map_err(|e| {
                ctx.device.destroy_buffer(fft_buffer, None);
                ctx.device.free_memory(fft_memory, None);
                VulkanError::vk("create_command_pool", e)
            })?
        };

        // SAFETY: ctx.device is valid.
        let fence = unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| {
                    ctx.device.destroy_command_pool(command_pool, None);
                    ctx.device.destroy_buffer(fft_buffer, None);
                    ctx.device.free_memory(fft_memory, None);
                    VulkanError::vk("create_fence", e)
                })?
        };

        let handles = HandleStorage::new(
            ctx.physical_device,
            ctx.device.handle(),
            ctx.queue,
            command_pool,
            fence,
            fft_buffer,
            size_bytes,
        );

        let mut inner = Box::new(PlanInner {
            ctx,
            app: unsafe { std::mem::zeroed() },
            handles,
            command_pool,
            fence,
            fft_buffer,
            fft_memory,
            element_count,
            size_bytes,
            desc,
        });

        // Build VkFFT configuration. Pointers reach into `inner.handles`,
        // whose address is stable because `inner` lives on the heap via Box.
        //
        // SAFETY: `VkFFTConfiguration` is a plain-old-data struct (bindgen
        // emits `#[derive(Default)]` via `derive_default(true)`). All pointer
        // fields we populate target the boxed `inner.handles`, so they stay
        // valid until the plan drops.
        let init_result = unsafe {
            let h = &mut inner.handles;
            let mut cfg: sys::VkFFTConfiguration = std::mem::zeroed();
            cfg.FFTdim = inner.desc.shape.rank() as u64;
            match inner.desc.shape {
                Shape::D1(n) => {
                    cfg.size[0] = n as u64;
                    cfg.numberBatches = inner.desc.batch as u64;
                }
                Shape::D2([nx, ny]) => {
                    cfg.size[0] = nx as u64;
                    cfg.size[1] = ny as u64;
                }
                Shape::D3([nx, ny, nz]) => {
                    cfg.size[0] = nx as u64;
                    cfg.size[1] = ny as u64;
                    cfg.size[2] = nz as u64;
                }
            }
            if matches!(T::PRECISION, Precision::F64) {
                cfg.doublePrecision = 1;
            }
            if inner.desc.normalize {
                cfg.normalize = 1;
            }

            cfg.physicalDevice = (&mut h.physical_device as *mut u64).cast();
            cfg.device = (&mut h.device as *mut u64).cast();
            cfg.queue = (&mut h.queue as *mut u64).cast();
            cfg.commandPool = (&mut h.command_pool as *mut u64).cast();
            cfg.fence = (&mut h.fence as *mut u64).cast();
            cfg.buffer = (&mut h.buffer as *mut u64).cast();
            cfg.bufferSize = &mut h.buffer_size as *mut u64;
            cfg.bufferNum = 1;

            sys::gpufft_vkfft_init(&mut inner.app as *mut _, cfg)
        };

        if init_result != 0 {
            // SAFETY: resources are ours and not yet released.
            unsafe {
                inner.ctx.device.destroy_fence(inner.fence, None);
                inner
                    .ctx
                    .device
                    .destroy_command_pool(inner.command_pool, None);
                inner.ctx.device.destroy_buffer(inner.fft_buffer, None);
                inner.ctx.device.free_memory(inner.fft_memory, None);
            }
            return Err(VulkanError::VkFft { code: init_result });
        }

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }
}

impl<T: Scalar> PlanOps<super::VulkanBackend, T> for VulkanPlan<T> {
    fn execute(
        &mut self,
        buffer: &mut VulkanBuffer<T>,
        direction: Direction,
    ) -> Result<(), VulkanError> {
        if buffer.len != self.inner.element_count {
            return Err(VulkanError::LengthMismatch {
                expected: self.inner.element_count,
                got: buffer.len,
            });
        }

        let ctx = self.inner.ctx.clone();
        copy_buffer(&ctx, buffer.buffer, self.inner.fft_buffer, self.inner.size_bytes)?;

        let code = append_and_execute(&self.inner, direction)?;
        if code != 0 {
            return Err(VulkanError::VkFft { code });
        }

        copy_buffer(&ctx, self.inner.fft_buffer, buffer.buffer, self.inner.size_bytes)?;
        Ok(())
    }
}

impl<T: Scalar> Drop for VulkanPlan<T> {
    fn drop(&mut self) {
        // Delete VkFFT resources before destroying the Vulkan objects they
        // reference. `handles` drops together with `inner`.
        //
        // SAFETY: app was initialised by `gpufft_vkfft_init`; deleteVkFFT is
        // the matching teardown. The Vulkan handles remain valid for the
        // duration of this drop.
        unsafe {
            sys::gpufft_vkfft_delete(&mut self.inner.app as *mut _);
            self.inner.ctx.device.device_wait_idle().ok();
            self.inner.ctx.device.destroy_fence(self.inner.fence, None);
            self.inner
                .ctx
                .device
                .destroy_command_pool(self.inner.command_pool, None);
            self.inner.ctx.device.destroy_buffer(self.inner.fft_buffer, None);
            self.inner.ctx.device.free_memory(self.inner.fft_memory, None);
        }
    }
}

fn validate_desc<T: Scalar>(desc: &PlanDesc) -> Result<(), VulkanError> {
    if !matches!(desc.transform, Transform::C2c) {
        return Err(VulkanError::UnsupportedTransform(desc.transform));
    }
    if !T::IS_COMPLEX {
        return Err(VulkanError::UnsupportedScalar(
            "v0.1 C2C transforms require a complex scalar (Complex32 or Complex64)",
        ));
    }
    if desc.batch == 0 {
        return Err(VulkanError::InvalidPlan("batch must be at least 1"));
    }
    if desc.batch > 1 && desc.shape.rank() > 1 {
        return Err(VulkanError::InvalidPlan(
            "batch > 1 is supported only for 1D shapes in v0.1",
        ));
    }
    Ok(())
}

fn copy_buffer(
    ctx: &VulkanContext,
    src: vk::Buffer,
    dst: vk::Buffer,
    size_bytes: u64,
) -> Result<(), VulkanError> {
    // SAFETY: see buffer.rs `copy_buffer_to_buffer`; this is the same recipe
    // but against the context-level transfer pool and fence.
    unsafe {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.transfer_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = ctx
            .device
            .allocate_command_buffers(&alloc)
            .map_err(|e| VulkanError::vk("allocate_command_buffers", e))?;
        let cmd = cmd_bufs[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.device
            .begin_command_buffer(cmd, &begin)
            .map_err(|e| VulkanError::vk("begin_command_buffer", e))?;

        let region = [vk::BufferCopy::default().size(size_bytes)];
        ctx.device.cmd_copy_buffer(cmd, src, dst, &region);
        ctx.device
            .end_command_buffer(cmd)
            .map_err(|e| VulkanError::vk("end_command_buffer", e))?;

        let submit = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
        ctx.device
            .reset_fences(&[ctx.transfer_fence])
            .map_err(|e| VulkanError::vk("reset_fences", e))?;
        ctx.device
            .queue_submit(ctx.queue, &submit, ctx.transfer_fence)
            .map_err(|e| VulkanError::vk("queue_submit", e))?;
        ctx.device
            .wait_for_fences(&[ctx.transfer_fence], true, u64::MAX)
            .map_err(|e| VulkanError::vk("wait_for_fences", e))?;

        ctx.device.free_command_buffers(ctx.transfer_pool, &cmd_bufs);
    }

    Ok(())
}

fn append_and_execute(inner: &PlanInner, direction: Direction) -> Result<i32, VulkanError> {
    let inverse = direction.as_int();

    // SAFETY: command buffer is allocated, recorded, submitted, waited on,
    // and freed within the same unsafe block. `gpufft_vkfft_append` takes
    // stable pointers into `launch_handles`, whose lifetime covers the
    // entire append-submit-wait sequence.
    unsafe {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(inner.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = inner
            .ctx
            .device
            .allocate_command_buffers(&alloc)
            .map_err(|e| VulkanError::vk("allocate_command_buffers", e))?;
        let cmd = cmd_bufs[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        inner
            .ctx
            .device
            .begin_command_buffer(cmd, &begin)
            .map_err(|e| VulkanError::vk("begin_command_buffer", e))?;

        // Stable storage for VkFFTLaunchParams pointer fields. Lives until
        // after end_command_buffer, which is when VkFFT finishes recording.
        let mut cmd_buf_raw = cmd.as_raw();
        let mut buf_handle = inner.fft_buffer.as_raw();

        let mut params: sys::VkFFTLaunchParams = std::mem::zeroed();
        params.commandBuffer = (&mut cmd_buf_raw as *mut u64).cast();
        params.buffer = (&mut buf_handle as *mut u64).cast();

        let code = sys::gpufft_vkfft_append(
            (&inner.app as *const sys::VkFFTApplication).cast_mut(),
            inverse,
            &mut params as *mut _,
        );

        inner
            .ctx
            .device
            .end_command_buffer(cmd)
            .map_err(|e| VulkanError::vk("end_command_buffer", e))?;

        if code != 0 {
            inner
                .ctx
                .device
                .free_command_buffers(inner.command_pool, &cmd_bufs);
            return Ok(code);
        }

        let submit = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
        inner
            .ctx
            .device
            .reset_fences(&[inner.fence])
            .map_err(|e| VulkanError::vk("reset_fences", e))?;
        inner
            .ctx
            .device
            .queue_submit(inner.ctx.queue, &submit, inner.fence)
            .map_err(|e| VulkanError::vk("queue_submit", e))?;
        inner
            .ctx
            .device
            .wait_for_fences(&[inner.fence], true, u64::MAX)
            .map_err(|e| VulkanError::vk("wait_for_fences", e))?;

        inner
            .ctx
            .device
            .free_command_buffers(inner.command_pool, &cmd_bufs);

        Ok(0)
    }
}
