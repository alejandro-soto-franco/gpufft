//! Vulkan-backed FFT buffer.
//!
//! Device-local memory allocated via ash; host transfers go through a
//! throwaway HOST_VISIBLE+HOST_COHERENT staging buffer.

use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;

use super::device::VulkanContext;
use super::error::VulkanError;
use crate::backend::BufferOps;
use crate::scalar::Scalar;

/// A typed GPU buffer in device-local Vulkan memory.
pub struct VulkanBuffer<T: Scalar> {
    pub(crate) ctx: Arc<VulkanContext>,
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) size_bytes: u64,
    pub(crate) len: usize,
    _marker: PhantomData<T>,
}

impl<T: Scalar> VulkanBuffer<T> {
    pub(crate) fn new(ctx: Arc<VulkanContext>, len: usize) -> Result<Self, VulkanError> {
        let size_bytes = (len * T::BYTES) as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let (buffer, memory, _) = ctx.allocate_buffer(
            size_bytes,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Ok(Self {
            ctx,
            buffer,
            memory,
            size_bytes,
            len,
            _marker: PhantomData,
        })
    }

    /// Raw Vulkan buffer handle.
    pub fn raw(&self) -> vk::Buffer {
        self.buffer
    }

    /// Buffer size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

impl<T: Scalar> BufferOps<super::VulkanBackend, T> for VulkanBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn write(&mut self, src: &[T]) -> Result<(), VulkanError> {
        if src.len() != self.len {
            return Err(VulkanError::LengthMismatch {
                expected: self.len,
                got: src.len(),
            });
        }
        staging_copy_in(&self.ctx, self.buffer, self.size_bytes, src)
    }

    fn read(&self, dst: &mut [T]) -> Result<(), VulkanError> {
        if dst.len() != self.len {
            return Err(VulkanError::LengthMismatch {
                expected: self.len,
                got: dst.len(),
            });
        }
        staging_copy_out(&self.ctx, self.buffer, self.size_bytes, dst)
    }
}

impl<T: Scalar> Drop for VulkanBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: buffer and memory were created by us and not yet destroyed.
        // The ctx Arc ensures the ash::Device outlives this call.
        unsafe {
            self.ctx.device.destroy_buffer(self.buffer, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

fn staging_copy_in<T: Scalar>(
    ctx: &VulkanContext,
    dst: vk::Buffer,
    size_bytes: u64,
    src: &[T],
) -> Result<(), VulkanError> {
    let (staging, staging_mem, _) = ctx.allocate_buffer(
        size_bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // SAFETY: staging_mem is HOST_VISIBLE and just allocated; mapping for the
    // full size is legal until unmap.
    unsafe {
        let ptr = ctx
            .device
            .map_memory(staging_mem, 0, size_bytes, vk::MemoryMapFlags::empty())
            .map_err(|e| {
                ctx.device.destroy_buffer(staging, None);
                ctx.device.free_memory(staging_mem, None);
                VulkanError::vk("map_memory", e)
            })?;
        std::ptr::copy_nonoverlapping(
            src.as_ptr() as *const u8,
            ptr as *mut u8,
            size_bytes as usize,
        );
        ctx.device.unmap_memory(staging_mem);
    }

    let result = copy_buffer_to_buffer(ctx, staging, dst, size_bytes);

    // SAFETY: staging + staging_mem are ours.
    unsafe {
        ctx.device.destroy_buffer(staging, None);
        ctx.device.free_memory(staging_mem, None);
    }

    result
}

fn staging_copy_out<T: Scalar>(
    ctx: &VulkanContext,
    src: vk::Buffer,
    size_bytes: u64,
    dst: &mut [T],
) -> Result<(), VulkanError> {
    let (staging, staging_mem, _) = ctx.allocate_buffer(
        size_bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let copy_result = copy_buffer_to_buffer(ctx, src, staging, size_bytes);
    if let Err(e) = copy_result {
        // SAFETY: staging + staging_mem are ours.
        unsafe {
            ctx.device.destroy_buffer(staging, None);
            ctx.device.free_memory(staging_mem, None);
        }
        return Err(e);
    }

    // SAFETY: staging_mem is HOST_VISIBLE.
    unsafe {
        let ptr = ctx
            .device
            .map_memory(staging_mem, 0, size_bytes, vk::MemoryMapFlags::empty())
            .map_err(|e| {
                ctx.device.destroy_buffer(staging, None);
                ctx.device.free_memory(staging_mem, None);
                VulkanError::vk("map_memory", e)
            })?;
        std::ptr::copy_nonoverlapping(
            ptr as *const u8,
            dst.as_mut_ptr() as *mut u8,
            size_bytes as usize,
        );
        ctx.device.unmap_memory(staging_mem);
    }

    // SAFETY: staging + staging_mem are ours.
    unsafe {
        ctx.device.destroy_buffer(staging, None);
        ctx.device.free_memory(staging_mem, None);
    }

    Ok(())
}

fn copy_buffer_to_buffer(
    ctx: &VulkanContext,
    src: vk::Buffer,
    dst: vk::Buffer,
    size_bytes: u64,
) -> Result<(), VulkanError> {
    // SAFETY: all Vulkan operations below are sequenced correctly: allocate
    // cmd buf, record begin/copy/end, submit with fence, wait, free.
    unsafe {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.transfer_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = ctx
            .device
            .allocate_command_buffers(&alloc_info)
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
