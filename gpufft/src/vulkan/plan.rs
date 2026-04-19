//! Vulkan FFT plan types wrapping VkFFT applications.
//!
//! # Execution strategy
//!
//! Each plan type owns a **persistent command buffer** that is reset and
//! re-recorded on every `execute`. All copy-in + VkFFT + copy-out work is
//! placed in a **single command buffer** with a **single submit + fence
//! wait** per call, eliminating 2 of the 3 fence waits per execute that
//! the previous design incurred.
//!
//! Additional per-transform notes:
//!
//! - **C2C**: truly zero-copy. `VkFFTLaunchParams.buffer` is overridden to
//!   point at the user's buffer; the plan's small scratch buffer only
//!   exists to satisfy VkFFT's init-time handle binding.
//! - **R2C / C2R**: in-place padded layout on a single internal buffer
//!   sized to hold either `n_outer * 2 * (innermost/2 + 1)` reals or the
//!   tight complex half-spectrum (same total bytes). The copy-in for R2C
//!   pads each innermost row from `innermost` reals to
//!   `2 * (innermost/2 + 1)` reals via a multi-region `vkCmdCopyBuffer`
//!   recorded into the same command buffer as VkFFTAppend; no extra
//!   submission. C2R is the symmetric reverse.
//!
//!   Zero-copy R2C/C2R via `isInputFormatted`+stride overrides was tried
//!   first but produced incorrect results on multi-dimensional shapes.
//!   Revisit in a follow-up once VkFFT's multi-dim formatted-input path
//!   is understood; in the meantime padded-in-place is correct and still
//!   faster than the previous 3-submission design.
//!
//! # Dimension ordering convention
//!
//! A [`Shape`] is the **real-space physical shape** in ndarray row-major
//! order (first axis slowest, last axis contiguous). VkFFT's `size[0]` is
//! the W axis, the fastest-varying (stride-1) dimension, so
//! `VkFFTConfiguration::size` is populated in **reverse**:
//! `size[0] = shape_last`. This matches cuFFT: `Shape::D3([nx, ny, nz])`
//! treats `nz` as contiguous and R2C output is `(nx, ny, nz/2+1)` on both
//! backends.
//!
//! # Lifetime invariant
//!
//! `VkFFTConfiguration` retains raw pointers to its handle fields for
//! later `VkFFTAppend` calls. All handle storage lives inside a boxed
//! `Inner` struct so the pointer targets are stable for the plan's
//! lifetime.

use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;
use gpufft_vulkan_sys as sys;

use super::buffer::VulkanBuffer;
use super::device::VulkanContext;
use super::error::VulkanError;
use super::kernels::StrideCopyKernel;
use crate::backend::{C2cPlanOps, C2rPlanOps, R2cPlanOps};
use crate::plan::{Direction, PlanDesc, Shape};
use crate::scalar::{Complex, Precision, Real};

// ============================================================
// Shared plumbing
// ============================================================

/// Stable storage for all Vulkan handles that VkFFT retains pointers into.
#[derive(Clone, Copy, Debug)]
struct Handles {
    physical_device: u64,
    device: u64,
    queue: u64,
    command_pool: u64,
    fence: u64,
    buffer: u64,
    buffer_size: u64,
}

impl Handles {
    fn new(
        ctx: &VulkanContext,
        command_pool: vk::CommandPool,
        fence: vk::Fence,
        buffer: vk::Buffer,
        buffer_size: u64,
    ) -> Self {
        Self {
            physical_device: ctx.physical_device.as_raw(),
            device: ctx.device.handle().as_raw(),
            queue: ctx.queue.as_raw(),
            command_pool: command_pool.as_raw(),
            fence: fence.as_raw(),
            buffer: buffer.as_raw(),
            buffer_size,
        }
    }
}

fn bind_cfg_handles(cfg: &mut sys::VkFFTConfiguration, h: &mut Handles) {
    cfg.physicalDevice = std::ptr::from_mut(&mut h.physical_device).cast();
    cfg.device = std::ptr::from_mut(&mut h.device).cast();
    cfg.queue = std::ptr::from_mut(&mut h.queue).cast();
    cfg.commandPool = std::ptr::from_mut(&mut h.command_pool).cast();
    cfg.fence = std::ptr::from_mut(&mut h.fence).cast();
    cfg.buffer = std::ptr::from_mut(&mut h.buffer).cast();
    cfg.bufferSize = std::ptr::from_mut(&mut h.buffer_size);
    cfg.bufferNum = 1;
}

fn set_cfg_size(cfg: &mut sys::VkFFTConfiguration, shape: &Shape) {
    cfg.FFTdim = shape.rank() as u64;
    match *shape {
        Shape::D1(n) => {
            cfg.size[0] = n as u64;
        }
        Shape::D2([a, b]) => {
            cfg.size[0] = b as u64;
            cfg.size[1] = a as u64;
        }
        Shape::D3([a, b, c]) => {
            cfg.size[0] = c as u64;
            cfg.size[1] = b as u64;
            cfg.size[2] = a as u64;
        }
    }
}

fn validate_desc_common(desc: &PlanDesc) -> Result<(), VulkanError> {
    if desc.batch == 0 {
        return Err(VulkanError::InvalidPlan("batch must be at least 1"));
    }
    if desc.batch > 1 && desc.shape.rank() > 1 {
        return Err(VulkanError::InvalidPlan(
            "batch > 1 is supported only for 1D shapes",
        ));
    }
    Ok(())
}

fn allocate_device_local_buffer(
    ctx: &VulkanContext,
    size_bytes: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory), VulkanError> {
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST;
    let (buffer, memory, _) =
        ctx.allocate_buffer(size_bytes, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    Ok((buffer, memory))
}

fn create_pool_and_fence(ctx: &VulkanContext) -> Result<(vk::CommandPool, vk::Fence), VulkanError> {
    // SAFETY: ctx.device is valid for its lifetime.
    let pool = unsafe {
        let ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        ctx.device
            .create_command_pool(&ci, None)
            .map_err(|e| VulkanError::vk("create_command_pool", e))?
    };
    // SAFETY: same.
    let fence = unsafe {
        match ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
        {
            Ok(f) => f,
            Err(e) => {
                ctx.device.destroy_command_pool(pool, None);
                return Err(VulkanError::vk("create_fence", e));
            }
        }
    };
    Ok((pool, fence))
}

fn allocate_persistent_cmd_buf(
    ctx: &VulkanContext,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, VulkanError> {
    // SAFETY: pool is valid.
    unsafe {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = ctx
            .device
            .allocate_command_buffers(&alloc)
            .map_err(|e| VulkanError::vk("allocate_command_buffers", e))?;
        Ok(cmd_bufs[0])
    }
}

/// Full read/write memory barrier around transfer <-> compute stages.
/// Recorded between each copy and the VkFFT dispatch so the driver sees
/// the dependency ordering inside a single command buffer.
fn record_full_barrier(ctx: &VulkanContext, cmd: vk::CommandBuffer) {
    // SAFETY: command buffer is in recording state.
    unsafe {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE,
            );
        ctx.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

fn submit_and_wait(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
) -> Result<(), VulkanError> {
    // SAFETY: cmd was recorded and ended; we submit and wait within this block.
    unsafe {
        let cmd_bufs = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
        ctx.device
            .reset_fences(&[fence])
            .map_err(|e| VulkanError::vk("reset_fences", e))?;
        ctx.device
            .queue_submit(ctx.queue, &submit, fence)
            .map_err(|e| VulkanError::vk("queue_submit", e))?;
        ctx.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| VulkanError::vk("wait_for_fences", e))?;
    }
    Ok(())
}

fn begin_persistent_cmd(ctx: &VulkanContext, cmd: vk::CommandBuffer) -> Result<(), VulkanError> {
    // SAFETY: the plan owns `cmd`; it is always in "executable" or
    // "initial" state at entry.
    unsafe {
        ctx.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| VulkanError::vk("reset_command_buffer", e))?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.device
            .begin_command_buffer(cmd, &begin)
            .map_err(|e| VulkanError::vk("begin_command_buffer", e))?;
    }
    Ok(())
}

fn end_cmd(ctx: &VulkanContext, cmd: vk::CommandBuffer) -> Result<(), VulkanError> {
    // SAFETY: cmd is in recording state.
    unsafe {
        ctx.device
            .end_command_buffer(cmd)
            .map_err(|e| VulkanError::vk("end_command_buffer", e))
    }
}

// ============================================================
// C2C (in-place, zero-copy, single submit)
// ============================================================

/// VkFFT-backed complex-to-complex in-place FFT plan.
pub struct VulkanC2cPlan<T: Complex> {
    inner: Box<C2cInner>,
    _marker: PhantomData<T>,
}

struct C2cInner {
    ctx: Arc<VulkanContext>,
    app: sys::VkFFTApplication,
    handles: Handles,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    /// Init-only scratch; never touched at runtime because VkFFTLaunchParams
    /// overrides `buffer` with the user's buffer handle.
    scratch_buffer: vk::Buffer,
    scratch_memory: vk::DeviceMemory,
    element_count: usize,
}

impl<T: Complex> VulkanC2cPlan<T> {
    pub(crate) fn new(ctx: Arc<VulkanContext>, desc: PlanDesc) -> Result<Self, VulkanError> {
        validate_desc_common(&desc)?;

        let element_count = (desc.shape.elements() * desc.batch as u64) as usize;
        let buffer_bytes = (element_count * T::BYTES) as u64;

        let (scratch_buffer, scratch_memory) = allocate_device_local_buffer(&ctx, buffer_bytes)?;
        let (command_pool, fence) = match create_pool_and_fence(&ctx) {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: scratch just allocated and not bound elsewhere.
                unsafe {
                    ctx.device.destroy_buffer(scratch_buffer, None);
                    ctx.device.free_memory(scratch_memory, None);
                }
                return Err(e);
            }
        };
        let command_buffer = match allocate_persistent_cmd_buf(&ctx, command_pool) {
            Ok(c) => c,
            Err(e) => {
                // SAFETY: freshly-created resources owned by us.
                unsafe {
                    ctx.device.destroy_fence(fence, None);
                    ctx.device.destroy_command_pool(command_pool, None);
                    ctx.device.destroy_buffer(scratch_buffer, None);
                    ctx.device.free_memory(scratch_memory, None);
                }
                return Err(e);
            }
        };

        let handles = Handles::new(&ctx, command_pool, fence, scratch_buffer, buffer_bytes);

        // SAFETY: POD zero is VkFFT's uninitialized-app contract.
        let zeroed_app: sys::VkFFTApplication = unsafe { std::mem::zeroed() };

        let mut inner = Box::new(C2cInner {
            ctx,
            app: zeroed_app,
            handles,
            command_pool,
            fence,
            command_buffer,
            scratch_buffer,
            scratch_memory,
            element_count,
        });

        // SAFETY: cfg pointers target boxed `inner.handles`; heap address stable.
        let init = unsafe {
            let h = &mut inner.handles;
            let mut cfg: sys::VkFFTConfiguration = std::mem::zeroed();
            set_cfg_size(&mut cfg, &desc.shape);
            if let Shape::D1(_) = desc.shape {
                cfg.numberBatches = desc.batch as u64;
            }
            if matches!(T::PRECISION, Precision::F64) {
                cfg.doublePrecision = 1;
            }
            if desc.normalize {
                cfg.normalize = 1;
            }
            bind_cfg_handles(&mut cfg, h);
            sys::gpufft_vkfft_init(std::ptr::from_mut(&mut inner.app), cfg)
        };

        if init != 0 {
            destroy_c2c_inner(&mut inner);
            return Err(VulkanError::VkFft { code: init });
        }

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }
}

impl<T: Complex> C2cPlanOps<super::VulkanBackend, T> for VulkanC2cPlan<T> {
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

        begin_persistent_cmd(&self.inner.ctx, self.inner.command_buffer)?;

        // SAFETY: cmd is recording; slots outlive record+submit+wait.
        unsafe {
            let mut cmd_buf_slot: u64 = self.inner.command_buffer.as_raw();
            let mut buffer_slot: u64 = buffer.buffer.as_raw();

            let mut params: sys::VkFFTLaunchParams = std::mem::zeroed();
            params.commandBuffer = std::ptr::from_mut(&mut cmd_buf_slot).cast();
            params.buffer = std::ptr::from_mut(&mut buffer_slot).cast();

            let code = sys::gpufft_vkfft_append(
                std::ptr::from_mut(&mut self.inner.app),
                direction.as_int(),
                std::ptr::from_mut(&mut params),
            );
            end_cmd(&self.inner.ctx, self.inner.command_buffer)?;
            if code != 0 {
                return Err(VulkanError::VkFft { code });
            }
        }

        submit_and_wait(&self.inner.ctx, self.inner.command_buffer, self.inner.fence)
    }
}

impl<T: Complex> Drop for VulkanC2cPlan<T> {
    fn drop(&mut self) {
        // SAFETY: app initialized, not yet destroyed.
        unsafe {
            sys::gpufft_vkfft_delete(std::ptr::from_mut(&mut self.inner.app));
        }
        destroy_c2c_inner(&mut self.inner);
    }
}

fn destroy_c2c_inner(inner: &mut C2cInner) {
    // SAFETY: all handles are owned by us and not yet destroyed.
    unsafe {
        let ctx = &inner.ctx;
        ctx.device.device_wait_idle().ok();
        ctx.device
            .free_command_buffers(inner.command_pool, &[inner.command_buffer]);
        ctx.device.destroy_fence(inner.fence, None);
        ctx.device.destroy_command_pool(inner.command_pool, None);
        ctx.device.destroy_buffer(inner.scratch_buffer, None);
        ctx.device.free_memory(inner.scratch_memory, None);
    }
}

// ============================================================
// R2C and C2R (padded in-place, fused single submission)
// ============================================================

/// Logical shape metadata for real-transform padding arithmetic.
#[derive(Clone, Copy, Debug)]
struct RealDims {
    /// Innermost (contiguous) axis length. Halved on the complex side.
    innermost: u64,
    /// Number of innermost rows in the volume (outer axes times batch).
    n_rows: u64,
    /// Padded real elements per row: `2 * (innermost / 2 + 1)`.
    padded_inner_reals: u64,
    /// Complex elements per row in the half-spectrum: `innermost / 2 + 1`.
    complex_inner: u64,
}

impl RealDims {
    fn of(shape: &Shape, batch: u32) -> Self {
        let (innermost, outer_product) = match shape {
            Shape::D1(n) => (*n as u64, 1u64),
            Shape::D2([a, b]) => (*b as u64, *a as u64),
            Shape::D3([a, b, c]) => (*c as u64, *a as u64 * *b as u64),
        };
        let complex_inner = innermost / 2 + 1;
        Self {
            innermost,
            n_rows: outer_product * batch as u64,
            padded_inner_reals: 2 * complex_inner,
            complex_inner,
        }
    }
}

struct RealPlanInner {
    ctx: Arc<VulkanContext>,
    app: sys::VkFFTApplication,
    handles: Handles,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    /// Single in-place VkFFT buffer sized to hold either the padded real
    /// layout (real space) or the tight complex half-spectrum (frequency
    /// space). Same total bytes, reinterpreted per direction.
    fft_buffer: vk::Buffer,
    fft_memory: vk::DeviceMemory,
    /// Stride-aware copy kernel: replaces the per-row `vkCmdCopyBuffer`
    /// multi-region dance for padding / stripping the innermost axis.
    stride_kernel: StrideCopyKernel,
    dims: RealDims,
    real_element_count: usize,
    complex_element_count: usize,
    elem_bytes: u64,
}

impl RealPlanInner {
    fn new<F: Real>(ctx: Arc<VulkanContext>, desc: PlanDesc) -> Result<Box<Self>, VulkanError> {
        validate_desc_common(&desc)?;

        let dims = RealDims::of(&desc.shape, desc.batch);
        let real_element_count = (desc.shape.elements() * desc.batch as u64) as usize;
        let complex_element_count =
            (desc.shape.complex_half_elements() * desc.batch as u64) as usize;
        let elem_bytes = F::BYTES as u64;
        let size_bytes = dims.n_rows * dims.padded_inner_reals * elem_bytes;

        let (fft_buffer, fft_memory) = allocate_device_local_buffer(&ctx, size_bytes)?;
        let (command_pool, fence) = match create_pool_and_fence(&ctx) {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: fresh allocations owned by us.
                unsafe {
                    ctx.device.destroy_buffer(fft_buffer, None);
                    ctx.device.free_memory(fft_memory, None);
                }
                return Err(e);
            }
        };
        let command_buffer = match allocate_persistent_cmd_buf(&ctx, command_pool) {
            Ok(c) => c,
            Err(e) => {
                // SAFETY: same.
                unsafe {
                    ctx.device.destroy_fence(fence, None);
                    ctx.device.destroy_command_pool(command_pool, None);
                    ctx.device.destroy_buffer(fft_buffer, None);
                    ctx.device.free_memory(fft_memory, None);
                }
                return Err(e);
            }
        };

        let stride_kernel = match StrideCopyKernel::new(ctx.clone()) {
            Ok(k) => k,
            Err(e) => {
                // SAFETY: same.
                unsafe {
                    ctx.device
                        .free_command_buffers(command_pool, &[command_buffer]);
                    ctx.device.destroy_fence(fence, None);
                    ctx.device.destroy_command_pool(command_pool, None);
                    ctx.device.destroy_buffer(fft_buffer, None);
                    ctx.device.free_memory(fft_memory, None);
                }
                return Err(e);
            }
        };

        let handles = Handles::new(&ctx, command_pool, fence, fft_buffer, size_bytes);

        // SAFETY: POD zero.
        let zeroed_app: sys::VkFFTApplication = unsafe { std::mem::zeroed() };

        let mut inner = Box::new(Self {
            ctx,
            app: zeroed_app,
            handles,
            command_pool,
            fence,
            command_buffer,
            fft_buffer,
            fft_memory,
            stride_kernel,
            dims,
            real_element_count,
            complex_element_count,
            elem_bytes,
        });

        // SAFETY: cfg pointers reach into boxed `inner.handles` with stable address.
        let init = unsafe {
            let h = &mut inner.handles;
            let mut cfg: sys::VkFFTConfiguration = std::mem::zeroed();
            set_cfg_size(&mut cfg, &desc.shape);
            if let Shape::D1(_) = desc.shape {
                cfg.numberBatches = desc.batch as u64;
            }
            if matches!(F::PRECISION, Precision::F64) {
                cfg.doublePrecision = 1;
            }
            if desc.normalize {
                cfg.normalize = 1;
            }
            cfg.performR2C = 1;
            bind_cfg_handles(&mut cfg, h);
            sys::gpufft_vkfft_init(std::ptr::from_mut(&mut inner.app), cfg)
        };

        if init != 0 {
            destroy_real_inner(&mut inner);
            return Err(VulkanError::VkFft { code: init });
        }

        Ok(inner)
    }
}

fn destroy_real_inner(inner: &mut RealPlanInner) {
    // SAFETY: all handles owned by us.
    unsafe {
        let ctx = &inner.ctx;
        ctx.device.device_wait_idle().ok();
        ctx.device
            .free_command_buffers(inner.command_pool, &[inner.command_buffer]);
        ctx.device.destroy_fence(inner.fence, None);
        ctx.device.destroy_command_pool(inner.command_pool, None);
        ctx.device.destroy_buffer(inner.fft_buffer, None);
        ctx.device.free_memory(inner.fft_memory, None);
    }
}

fn record_single_copy(
    ctx: &VulkanContext,
    cmd: vk::CommandBuffer,
    src: vk::Buffer,
    dst: vk::Buffer,
    size_bytes: u64,
) {
    let region = [vk::BufferCopy::default().size(size_bytes)];
    // SAFETY: cmd is recording.
    unsafe {
        ctx.device.cmd_copy_buffer(cmd, src, dst, &region);
    }
}

fn record_vkfft_append(
    app: *mut sys::VkFFTApplication,
    cmd: vk::CommandBuffer,
    buffer: vk::Buffer,
    direction: Direction,
) -> i32 {
    // SAFETY: stack-local slots live through the recording call; VkFFT
    // copies handle values out of the pointer targets internally.
    unsafe {
        let mut cmd_slot: u64 = cmd.as_raw();
        let mut buf_slot: u64 = buffer.as_raw();
        let mut params: sys::VkFFTLaunchParams = std::mem::zeroed();
        params.commandBuffer = std::ptr::from_mut(&mut cmd_slot).cast();
        params.buffer = std::ptr::from_mut(&mut buf_slot).cast();
        sys::gpufft_vkfft_append(app, direction.as_int(), std::ptr::from_mut(&mut params))
    }
}

// ---------- R2C ----------

/// VkFFT-backed real-to-complex forward plan.
pub struct VulkanR2cPlan<F: Real> {
    inner: Box<RealPlanInner>,
    _marker: PhantomData<F>,
}

impl<F: Real> VulkanR2cPlan<F> {
    pub(crate) fn new(ctx: Arc<VulkanContext>, desc: PlanDesc) -> Result<Self, VulkanError> {
        let inner = RealPlanInner::new::<F>(ctx, desc)?;
        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }
}

impl<F: Real> R2cPlanOps<super::VulkanBackend, F> for VulkanR2cPlan<F> {
    fn execute(
        &mut self,
        input: &VulkanBuffer<F>,
        output: &mut VulkanBuffer<F::Complex>,
    ) -> Result<(), VulkanError> {
        if input.len != self.inner.real_element_count {
            return Err(VulkanError::LengthMismatch {
                expected: self.inner.real_element_count,
                got: input.len,
            });
        }
        if output.len != self.inner.complex_element_count {
            return Err(VulkanError::LengthMismatch {
                expected: self.inner.complex_element_count,
                got: output.len,
            });
        }

        let dims = self.inner.dims;
        let elem_bytes = self.inner.elem_bytes;
        let elem_uints = (elem_bytes / 4) as u32;
        let real_bytes = (self.inner.real_element_count as u64) * elem_bytes;
        let padded_total_bytes = dims.n_rows * dims.padded_inner_reals * elem_bytes;
        let complex_bytes = dims.n_rows * dims.complex_inner * 2 * elem_bytes;

        // Update descriptor set before recording: src=user real, dst=padded fft buffer.
        self.inner.stride_kernel.update_descriptor(
            input.buffer,
            real_bytes,
            self.inner.fft_buffer,
            padded_total_bytes,
        );

        begin_persistent_cmd(&self.inner.ctx, self.inner.command_buffer)?;

        // Compute-shader padder: tight reals -> padded reals.
        let row_uints = (dims.innermost as u32) * elem_uints;
        let src_stride_uints = (dims.innermost as u32) * elem_uints;
        let dst_stride_uints = (dims.padded_inner_reals as u32) * elem_uints;
        self.inner.stride_kernel.record_dispatch(
            self.inner.command_buffer,
            row_uints,
            src_stride_uints,
            dst_stride_uints,
            dims.n_rows as u32,
        );
        record_full_barrier(&self.inner.ctx, self.inner.command_buffer);

        // VkFFT R2C in-place on fft_buffer.
        let code = record_vkfft_append(
            std::ptr::from_mut(&mut self.inner.app),
            self.inner.command_buffer,
            self.inner.fft_buffer,
            Direction::Forward,
        );
        if code != 0 {
            end_cmd(&self.inner.ctx, self.inner.command_buffer)?;
            return Err(VulkanError::VkFft { code });
        }
        record_full_barrier(&self.inner.ctx, self.inner.command_buffer);

        // Copy tight complex half-spectrum out to user's complex buffer.
        record_single_copy(
            &self.inner.ctx,
            self.inner.command_buffer,
            self.inner.fft_buffer,
            output.buffer,
            complex_bytes,
        );

        end_cmd(&self.inner.ctx, self.inner.command_buffer)?;
        submit_and_wait(&self.inner.ctx, self.inner.command_buffer, self.inner.fence)
    }
}

impl<F: Real> Drop for VulkanR2cPlan<F> {
    fn drop(&mut self) {
        // SAFETY: app initialized, not yet destroyed.
        unsafe {
            sys::gpufft_vkfft_delete(std::ptr::from_mut(&mut self.inner.app));
        }
        destroy_real_inner(&mut self.inner);
    }
}

// ---------- C2R ----------

/// VkFFT-backed complex-to-real inverse plan.
pub struct VulkanC2rPlan<F: Real> {
    inner: Box<RealPlanInner>,
    _marker: PhantomData<F>,
}

impl<F: Real> VulkanC2rPlan<F> {
    pub(crate) fn new(ctx: Arc<VulkanContext>, desc: PlanDesc) -> Result<Self, VulkanError> {
        let inner = RealPlanInner::new::<F>(ctx, desc)?;
        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }
}

impl<F: Real> C2rPlanOps<super::VulkanBackend, F> for VulkanC2rPlan<F> {
    fn execute(
        &mut self,
        input: &VulkanBuffer<F::Complex>,
        output: &mut VulkanBuffer<F>,
    ) -> Result<(), VulkanError> {
        if input.len != self.inner.complex_element_count {
            return Err(VulkanError::LengthMismatch {
                expected: self.inner.complex_element_count,
                got: input.len,
            });
        }
        if output.len != self.inner.real_element_count {
            return Err(VulkanError::LengthMismatch {
                expected: self.inner.real_element_count,
                got: output.len,
            });
        }

        let dims = self.inner.dims;
        let elem_bytes = self.inner.elem_bytes;
        let elem_uints = (elem_bytes / 4) as u32;
        let real_bytes = (self.inner.real_element_count as u64) * elem_bytes;
        let padded_total_bytes = dims.n_rows * dims.padded_inner_reals * elem_bytes;
        let complex_bytes = dims.n_rows * dims.complex_inner * 2 * elem_bytes;

        // Update descriptor set before recording: src=padded fft_buffer, dst=user real.
        self.inner.stride_kernel.update_descriptor(
            self.inner.fft_buffer,
            padded_total_bytes,
            output.buffer,
            real_bytes,
        );

        begin_persistent_cmd(&self.inner.ctx, self.inner.command_buffer)?;

        // Copy tight complex input into internal buffer (interpreted as complex).
        record_single_copy(
            &self.inner.ctx,
            self.inner.command_buffer,
            input.buffer,
            self.inner.fft_buffer,
            complex_bytes,
        );
        record_full_barrier(&self.inner.ctx, self.inner.command_buffer);

        // VkFFT C2R in-place.
        let code = record_vkfft_append(
            std::ptr::from_mut(&mut self.inner.app),
            self.inner.command_buffer,
            self.inner.fft_buffer,
            Direction::Inverse,
        );
        if code != 0 {
            end_cmd(&self.inner.ctx, self.inner.command_buffer)?;
            return Err(VulkanError::VkFft { code });
        }
        record_full_barrier(&self.inner.ctx, self.inner.command_buffer);

        // Compute-shader stripper: padded reals -> tight reals.
        let row_uints = (dims.innermost as u32) * elem_uints;
        let src_stride_uints = (dims.padded_inner_reals as u32) * elem_uints;
        let dst_stride_uints = (dims.innermost as u32) * elem_uints;
        self.inner.stride_kernel.record_dispatch(
            self.inner.command_buffer,
            row_uints,
            src_stride_uints,
            dst_stride_uints,
            dims.n_rows as u32,
        );

        end_cmd(&self.inner.ctx, self.inner.command_buffer)?;
        submit_and_wait(&self.inner.ctx, self.inner.command_buffer, self.inner.fence)
    }
}

impl<F: Real> Drop for VulkanC2rPlan<F> {
    fn drop(&mut self) {
        // SAFETY: app initialized, not yet destroyed.
        unsafe {
            sys::gpufft_vkfft_delete(std::ptr::from_mut(&mut self.inner.app));
        }
        destroy_real_inner(&mut self.inner);
    }
}
