//! Vulkan FFT plan types wrapping VkFFT applications.
//!
//! # Convention
//!
//! Throughout this module a [`Shape`] is the **real-space physical shape**
//! in the same ordering as an `ndarray::Array3` (first axis slowest, last
//! axis contiguous in row-major memory). VkFFT uses the opposite
//! orientation: its `size[0]` is the W axis, which is the fastest-varying
//! (stride-1) dimension. We therefore populate `VkFFTConfiguration::size`
//! in **reverse** order: `size[0] = shape_last`, `size[rank-1] = shape_first`.
//!
//! This matches the cuFFT backend: for a `Shape::D3([nx, ny, nz])` the
//! contiguous axis is `nz`, and R2C output is `(nx, ny, nz/2+1)` in both
//! backends.
//!
//! # Plan types
//!
//! - [`VulkanC2cPlan`]: in-place complex-to-complex, dispatched forward or
//!   inverse at `execute` time.
//! - [`VulkanR2cPlan`] / [`VulkanC2rPlan`]: in-place R2C / C2R on a single
//!   internal buffer sized to hold either the padded real layout or the
//!   tight complex half-spectrum. Upload pads each innermost row from
//!   `innermost` reals to `2 * (innermost / 2 + 1)` reals; download strips
//!   the padding in the C2R direction and is contiguous in the R2C
//!   direction.
//!
//! # Lifetime invariant
//!
//! `VkFFTConfiguration` retains pointers to its handle fields for later
//! `VkFFTAppend` calls. All handle storage therefore lives inside a boxed
//! `Inner` struct so the pointer targets are stable for the plan's
//! lifetime. Stack-local handle copies would dangle after init and cause a
//! segfault in `vkUpdateDescriptorSets`.

use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;
use gpufft_vulkan_sys as sys;

use super::buffer::VulkanBuffer;
use super::device::VulkanContext;
use super::error::VulkanError;
use super::handles::HandleStorage;
use crate::backend::{C2cPlanOps, C2rPlanOps, R2cPlanOps};
use crate::plan::{Direction, PlanDesc, Shape};
use crate::scalar::{Complex, Precision, Real};

// ---------- shape helpers ----------

/// Logical shape metadata needed for both R2C and C2R plans.
#[derive(Clone, Copy, Debug)]
struct ShapeDims {
    /// Contiguous (innermost) axis length. Halved on the complex side.
    innermost: u64,
    /// Number of independent innermost rows in the volume (= product of
    /// outer axes times `batch`).
    n_rows: u64,
    /// `2 * (innermost / 2 + 1)`: real elements per row in the padded
    /// layout, equivalent to `complex_half * 2`.
    padded_inner_reals: u64,
    /// `innermost / 2 + 1`: complex elements per row in the half-spectrum.
    complex_inner: u64,
}

impl ShapeDims {
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

/// Populate `cfg.size[i]` in VkFFT's innermost-first order from an
/// ndarray-style shape.
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

// ============================================================
// C2C
// ============================================================

/// VkFFT-backed complex-to-complex in-place FFT plan.
pub struct VulkanC2cPlan<T: Complex> {
    inner: Box<C2cInner>,
    _marker: PhantomData<T>,
}

struct C2cInner {
    ctx: Arc<VulkanContext>,
    app: sys::VkFFTApplication,
    handles: HandleStorage,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    fft_buffer: vk::Buffer,
    fft_memory: vk::DeviceMemory,
    element_count: usize,
    size_bytes: u64,
}

impl<T: Complex> VulkanC2cPlan<T> {
    pub(crate) fn new(ctx: Arc<VulkanContext>, desc: PlanDesc) -> Result<Self, VulkanError> {
        validate_desc_common(&desc)?;

        let element_count = (desc.shape.elements() * desc.batch as u64) as usize;
        let size_bytes = (element_count * T::BYTES) as u64;

        let (fft_buffer, fft_memory) = allocate_fft_buffer(&ctx, size_bytes)?;
        let (command_pool, fence) = match create_pool_and_fence(&ctx) {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: we just allocated `fft_buffer` and `fft_memory` and
                // neither has been bound to anything else.
                unsafe {
                    ctx.device.destroy_buffer(fft_buffer, None);
                    ctx.device.free_memory(fft_memory, None);
                }
                return Err(e);
            }
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

        // SAFETY: VkFFTApplication is POD; zero is the uninitialized state
        // expected by `initializeVkFFT`.
        let zeroed_app: sys::VkFFTApplication = unsafe { std::mem::zeroed() };

        let mut inner = Box::new(C2cInner {
            ctx,
            app: zeroed_app,
            handles,
            command_pool,
            fence,
            fft_buffer,
            fft_memory,
            element_count,
            size_bytes,
        });

        // SAFETY: all pointers into `handles` remain valid for `inner`'s
        // lifetime; `inner` is boxed so its heap address does not move.
        let init = unsafe {
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
            bind_cfg_handles(&mut cfg, &mut inner.handles);

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

        let ctx = self.inner.ctx.clone();
        copy_buffer_contiguous(
            &ctx,
            buffer.buffer,
            self.inner.fft_buffer,
            self.inner.size_bytes,
        )?;

        let code = append_c2c(&self.inner, direction)?;
        if code != 0 {
            return Err(VulkanError::VkFft { code });
        }

        copy_buffer_contiguous(
            &ctx,
            self.inner.fft_buffer,
            buffer.buffer,
            self.inner.size_bytes,
        )?;
        Ok(())
    }
}

impl<T: Complex> Drop for VulkanC2cPlan<T> {
    fn drop(&mut self) {
        // SAFETY: app was initialized by gpufft_vkfft_init; handles are ours
        // and have not been destroyed.
        unsafe {
            sys::gpufft_vkfft_delete(std::ptr::from_mut(&mut self.inner.app));
        }
        destroy_c2c_inner(&mut self.inner);
    }
}

fn destroy_c2c_inner(inner: &mut C2cInner) {
    // SAFETY: all four Vulkan handles are owned by us and valid.
    unsafe {
        let ctx = &inner.ctx;
        ctx.device.device_wait_idle().ok();
        ctx.device.destroy_fence(inner.fence, None);
        ctx.device.destroy_command_pool(inner.command_pool, None);
        ctx.device.destroy_buffer(inner.fft_buffer, None);
        ctx.device.free_memory(inner.fft_memory, None);
    }
}

fn append_c2c(inner: &C2cInner, direction: Direction) -> Result<i32, VulkanError> {
    record_and_submit(
        &inner.ctx,
        inner.command_pool,
        inner.fence,
        inner.fft_buffer,
        |app_ptr, cmd_buf_raw_slot, buf_handle_slot| {
            // SAFETY: same as `append_real_transform`; all pointers target
            // stack slots that outlive the append.
            unsafe {
                let mut params: sys::VkFFTLaunchParams = std::mem::zeroed();
                params.commandBuffer = std::ptr::from_mut(cmd_buf_raw_slot).cast();
                params.buffer = std::ptr::from_mut(buf_handle_slot).cast();
                sys::gpufft_vkfft_append(
                    app_ptr,
                    direction.as_int(),
                    std::ptr::from_mut(&mut params),
                )
            }
        },
        std::ptr::from_ref(&inner.app).cast_mut(),
    )
}

// ============================================================
// R2C
// ============================================================

/// VkFFT-backed real-to-complex forward plan.
pub struct VulkanR2cPlan<F: Real> {
    inner: Box<RealPlanInner>,
    _marker: PhantomData<F>,
}

/// VkFFT-backed complex-to-real inverse plan.
pub struct VulkanC2rPlan<F: Real> {
    inner: Box<RealPlanInner>,
    _marker: PhantomData<F>,
}

/// Inner state shared by both R2C and C2R plans. The plans differ only in
/// which direction data flows through the internal padded buffer.
struct RealPlanInner {
    ctx: Arc<VulkanContext>,
    app: sys::VkFFTApplication,
    handles: HandleStorage,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    /// Single in-place VkFFT buffer: holds either the padded real layout
    /// or the tight complex half-spectrum (same total bytes).
    fft_buffer: vk::Buffer,
    fft_memory: vk::DeviceMemory,
    dims: ShapeDims,
    real_element_count: usize,
    complex_element_count: usize,
    elem_bytes: u64,
}

impl RealPlanInner {
    fn new<F: Real>(ctx: Arc<VulkanContext>, desc: PlanDesc) -> Result<Box<Self>, VulkanError> {
        validate_desc_common(&desc)?;

        let dims = ShapeDims::of(&desc.shape, desc.batch);
        let real_element_count = (desc.shape.elements() * desc.batch as u64) as usize;
        let complex_element_count =
            (desc.shape.complex_half_elements() * desc.batch as u64) as usize;
        let elem_bytes = F::BYTES as u64;
        let size_bytes = dims.n_rows * dims.padded_inner_reals * elem_bytes;

        let (fft_buffer, fft_memory) = allocate_fft_buffer(&ctx, size_bytes)?;
        let (command_pool, fence) = match create_pool_and_fence(&ctx) {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: buffer and memory were just allocated and are otherwise untouched.
                unsafe {
                    ctx.device.destroy_buffer(fft_buffer, None);
                    ctx.device.free_memory(fft_memory, None);
                }
                return Err(e);
            }
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

        // SAFETY: VkFFTApplication is POD; zero is the uninitialized state.
        let zeroed_app: sys::VkFFTApplication = unsafe { std::mem::zeroed() };

        let mut inner = Box::new(Self {
            ctx,
            app: zeroed_app,
            handles,
            command_pool,
            fence,
            fft_buffer,
            fft_memory,
            dims,
            real_element_count,
            complex_element_count,
            elem_bytes,
        });

        // SAFETY: pointers bind into `inner.handles`, whose heap address is
        // stable for the plan's lifetime.
        let init = unsafe {
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
            bind_cfg_handles(&mut cfg, &mut inner.handles);

            sys::gpufft_vkfft_init(std::ptr::from_mut(&mut inner.app), cfg)
        };

        if init != 0 {
            destroy_real_inner(&mut inner);
            return Err(VulkanError::VkFft { code: init });
        }

        Ok(inner)
    }
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

        // Upload real input into the padded layout (inner row stride =
        // padded_inner_reals).
        copy_rows_strided(
            &self.inner.ctx,
            input.buffer,
            self.inner.fft_buffer,
            self.inner.dims.n_rows,
            self.inner.dims.innermost * self.inner.elem_bytes,
            self.inner.dims.padded_inner_reals * self.inner.elem_bytes,
            StridedDirection::PadInput,
        )?;

        let code = append_real(&self.inner, Direction::Forward)?;
        if code != 0 {
            return Err(VulkanError::VkFft { code });
        }

        // Tight complex output: single contiguous copy of
        // n_rows * complex_inner * sizeof(complex) bytes.
        let complex_bytes =
            self.inner.dims.n_rows * self.inner.dims.complex_inner * 2 * self.inner.elem_bytes;
        copy_buffer_contiguous(
            &self.inner.ctx,
            self.inner.fft_buffer,
            output.buffer,
            complex_bytes,
        )?;
        Ok(())
    }
}

impl<F: Real> Drop for VulkanR2cPlan<F> {
    fn drop(&mut self) {
        // SAFETY: app was initialized and has not been destroyed.
        unsafe {
            sys::gpufft_vkfft_delete(std::ptr::from_mut(&mut self.inner.app));
        }
        destroy_real_inner(&mut self.inner);
    }
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

        // Complex input lives tightly in the internal buffer's first
        // `complex_bytes` bytes.
        let complex_bytes =
            self.inner.dims.n_rows * self.inner.dims.complex_inner * 2 * self.inner.elem_bytes;
        copy_buffer_contiguous(
            &self.inner.ctx,
            input.buffer,
            self.inner.fft_buffer,
            complex_bytes,
        )?;

        let code = append_real(&self.inner, Direction::Inverse)?;
        if code != 0 {
            return Err(VulkanError::VkFft { code });
        }

        // Strip padding when writing back to the user's tight real buffer.
        copy_rows_strided(
            &self.inner.ctx,
            self.inner.fft_buffer,
            output.buffer,
            self.inner.dims.n_rows,
            self.inner.dims.innermost * self.inner.elem_bytes,
            self.inner.dims.padded_inner_reals * self.inner.elem_bytes,
            StridedDirection::StripOutput,
        )?;
        Ok(())
    }
}

impl<F: Real> Drop for VulkanC2rPlan<F> {
    fn drop(&mut self) {
        // SAFETY: app was initialized and has not been destroyed.
        unsafe {
            sys::gpufft_vkfft_delete(std::ptr::from_mut(&mut self.inner.app));
        }
        destroy_real_inner(&mut self.inner);
    }
}

fn destroy_real_inner(inner: &mut RealPlanInner) {
    // SAFETY: all handles are owned by us and valid.
    unsafe {
        let ctx = &inner.ctx;
        ctx.device.device_wait_idle().ok();
        ctx.device.destroy_fence(inner.fence, None);
        ctx.device.destroy_command_pool(inner.command_pool, None);
        ctx.device.destroy_buffer(inner.fft_buffer, None);
        ctx.device.free_memory(inner.fft_memory, None);
    }
}

fn append_real(inner: &RealPlanInner, direction: Direction) -> Result<i32, VulkanError> {
    record_and_submit(
        &inner.ctx,
        inner.command_pool,
        inner.fence,
        inner.fft_buffer,
        |app_ptr, cmd_buf_raw_slot, buf_handle_slot| {
            // SAFETY: pointers target stack slots that outlive the call.
            unsafe {
                let mut params: sys::VkFFTLaunchParams = std::mem::zeroed();
                params.commandBuffer = std::ptr::from_mut(cmd_buf_raw_slot).cast();
                params.buffer = std::ptr::from_mut(buf_handle_slot).cast();
                sys::gpufft_vkfft_append(
                    app_ptr,
                    direction.as_int(),
                    std::ptr::from_mut(&mut params),
                )
            }
        },
        std::ptr::from_ref(&inner.app).cast_mut(),
    )
}

// ============================================================
// Shared helpers
// ============================================================

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

fn allocate_fft_buffer(
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
    // SAFETY: ctx.device is valid.
    let pool = unsafe {
        let ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        ctx.device
            .create_command_pool(&ci, None)
            .map_err(|e| VulkanError::vk("create_command_pool", e))?
    };
    // SAFETY: ctx.device is valid.
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

fn bind_cfg_handles(cfg: &mut sys::VkFFTConfiguration, h: &mut HandleStorage) {
    cfg.physicalDevice = std::ptr::from_mut(&mut h.physical_device).cast();
    cfg.device = std::ptr::from_mut(&mut h.device).cast();
    cfg.queue = std::ptr::from_mut(&mut h.queue).cast();
    cfg.commandPool = std::ptr::from_mut(&mut h.command_pool).cast();
    cfg.fence = std::ptr::from_mut(&mut h.fence).cast();
    cfg.buffer = std::ptr::from_mut(&mut h.buffer).cast();
    cfg.bufferSize = std::ptr::from_mut(&mut h.buffer_size);
    cfg.bufferNum = 1;
}

fn copy_buffer_contiguous(
    ctx: &VulkanContext,
    src: vk::Buffer,
    dst: vk::Buffer,
    size_bytes: u64,
) -> Result<(), VulkanError> {
    // SAFETY: single-region copy; command buffer lifecycle is fully
    // sequenced within this block.
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

        ctx.device
            .free_command_buffers(ctx.transfer_pool, &cmd_bufs);
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum StridedDirection {
    /// Pad: tight src rows of `row_bytes` → padded dst slots of `padded_bytes`.
    PadInput,
    /// Strip: padded src slots of `padded_bytes` → tight dst rows of `row_bytes`.
    StripOutput,
}

fn copy_rows_strided(
    ctx: &VulkanContext,
    src: vk::Buffer,
    dst: vk::Buffer,
    n_rows: u64,
    row_bytes: u64,
    padded_bytes: u64,
    direction: StridedDirection,
) -> Result<(), VulkanError> {
    let mut regions: Vec<vk::BufferCopy> = Vec::with_capacity(n_rows as usize);
    for row in 0..n_rows {
        let (src_offset, dst_offset) = match direction {
            StridedDirection::PadInput => (row * row_bytes, row * padded_bytes),
            StridedDirection::StripOutput => (row * padded_bytes, row * row_bytes),
        };
        regions.push(
            vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(row_bytes),
        );
    }

    // SAFETY: all regions index within src and dst by construction.
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

        ctx.device.cmd_copy_buffer(cmd, src, dst, &regions);

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

        ctx.device
            .free_command_buffers(ctx.transfer_pool, &cmd_bufs);
    }

    Ok(())
}

/// Run a `VkFFTAppend`-style closure inside a one-shot command buffer.
///
/// The closure receives the application pointer and two stack slots
/// (`cmd_buf_raw`, `buf_handle`) that it can wire into a
/// [`sys::VkFFTLaunchParams`]. Their addresses remain valid for the full
/// record + submit + wait lifecycle.
fn record_and_submit<F>(
    ctx: &VulkanContext,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    buffer: vk::Buffer,
    append: F,
    app_ptr: *mut sys::VkFFTApplication,
) -> Result<i32, VulkanError>
where
    F: FnOnce(*mut sys::VkFFTApplication, &mut u64, &mut u64) -> i32,
{
    // SAFETY: standard command buffer lifecycle sequenced within this block.
    unsafe {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
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

        let mut cmd_buf_raw = cmd.as_raw();
        let mut buf_handle = buffer.as_raw();
        let code = append(app_ptr, &mut cmd_buf_raw, &mut buf_handle);

        ctx.device
            .end_command_buffer(cmd)
            .map_err(|e| VulkanError::vk("end_command_buffer", e))?;

        if code != 0 {
            ctx.device.free_command_buffers(command_pool, &cmd_bufs);
            return Ok(code);
        }

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

        ctx.device.free_command_buffers(command_pool, &cmd_bufs);

        Ok(0)
    }
}
