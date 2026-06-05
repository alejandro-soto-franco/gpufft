//! Device-local FFT-ready memory shared between Vulkan and CUDA.
//!
//! Like [`SharedMemory`](super::SharedMemory) but `DEVICE_LOCAL` instead of
//! host-visible, so it is usable as the storage buffer for both VkFFT and
//! cuFFT plans. Host transfers go through CUDA's `cudaMemcpy` against the raw
//! device pointer. No Vulkan staging buffer is needed.
//!
//! Pair this with `VulkanC2cPlan::execute_shared` and
//! `CudaC2cPlan::execute_shared` (Tasks 6 + 7) to run forward and inverse
//! transforms on the exact same physical allocation, on either backend,
//! without copies.

use std::ffi::c_void;
use std::os::unix::io::{FromRawFd, OwnedFd};

use ash::khr::external_memory_fd;
use ash::vk;
use gpufft_cuda_sys as sys;
use num_complex::Complex32;

use super::SharedMemoryError;

/// FFT-ready memory addressable from both Vulkan and CUDA.
///
/// Backed by `DEVICE_LOCAL` Vulkan memory exported as an `OPAQUE_FD` handle
/// and imported into CUDA via `cudaImportExternalMemory`. Both APIs address
/// the same physical bytes, with no host roundtrip and no staging buffer.
///
/// The Vulkan side exposes a `VkBuffer` with `STORAGE_BUFFER | TRANSFER_SRC |
/// TRANSFER_DST` usage, ready for VkFFT. The CUDA side exposes a
/// `*mut c_void` device pointer, ready for cuFFT.
pub struct SharedFftBuffer {
    /// The Vulkan buffer handle backed by `vk_memory`.
    vk_buffer: vk::Buffer,
    /// The exportable Vulkan device memory.
    vk_memory: vk::DeviceMemory,
    /// Cloned ash logical device; needed for teardown.
    ash_device: ash::Device,
    /// Number of `Complex32` elements this buffer holds.
    len_complex: usize,
    /// CUDA external-memory handle (owns the fd on the CUDA side).
    ext_mem_handle: sys::cudaExternalMemory_t,
    /// CUDA device pointer into the imported memory (`cudaFree` on drop).
    device_ptr: *mut c_void,
}

// SAFETY: The raw pointers `ext_mem_handle` and `device_ptr` are CUDA/Vulkan
// opaque handles. They are never dereferenced from Rust and only passed back
// to the respective driver APIs. Ownership is with `SharedFftBuffer`; drivers
// are thread-safe for handle destruction.
unsafe impl Send for SharedFftBuffer {}
unsafe impl Sync for SharedFftBuffer {}

impl SharedFftBuffer {
    /// Allocate `len_complex` Complex32 elements as `DEVICE_LOCAL` Vulkan
    /// memory, export it as an `OPAQUE_FD` handle, and import it into the
    /// CUDA context so both backends share the same physical bytes.
    pub fn new(
        vk_dev: &crate::vulkan::VulkanDevice,
        cuda_dev: &crate::cuda::CudaDevice,
        len_complex: usize,
    ) -> Result<Self, SharedMemoryError> {
        // TODO(t-followup): UUID-gated same-GPU check (parity with
        // cartan-gpu's check_same_gpu). UUID accessors aren't ported yet;
        // production use is gated by user discipline (matches cartan-gpu's own
        // doc warning "The minimal demo path here doesn't check the
        // adapter↔device UUID; production use should.").

        // Bind CUDA to the correct device before any runtime-API call.
        cuda_dev
            .make_current()
            .map_err(|e| SharedMemoryError::Cuda(format!("{e:?}")))?;

        let size_bytes = (len_complex * core::mem::size_of::<Complex32>()) as u64;

        // 1. Raw ash::Device + ash::Instance from VulkanDevice.
        let handles = vk_dev.raw_handles();
        let ash_device = handles.device.clone();
        let ash_instance = handles.instance.clone();
        let phys_dev = handles.physical_device; // already ash::vk::PhysicalDevice

        // 2. VkBuffer flagged for OPAQUE_FD export, usable as STORAGE_BUFFER
        //    and TRANSFER src/dst (VkFFT requires STORAGE_BUFFER).
        let vk_buffer = {
            let mut external_buf_info = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            let buf_info = vk::BufferCreateInfo::default()
                .size(size_bytes)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external_buf_info);
            // SAFETY: ash_device is a valid logical device for its lifetime.
            unsafe {
                ash_device.create_buffer(&buf_info, None).map_err(|e| {
                    SharedMemoryError::Vulkan(format!("create_buffer (shared-fft): {e:?}"))
                })?
            }
        };

        // 3. Memory requirements + DEVICE_LOCAL memory type index.
        // SAFETY: vk_buffer was just created above and is valid.
        let mem_req = unsafe { ash_device.get_buffer_memory_requirements(vk_buffer) };

        // SAFETY: phys_dev and ash_instance are valid for the device lifetime.
        let mem_props = unsafe { ash_instance.get_physical_device_memory_properties(phys_dev) };

        let mem_type_idx = (0..mem_props.memory_type_count)
            .find(|&i| {
                let supported = (mem_req.memory_type_bits & (1 << i)) != 0;
                let device_local = mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
                supported && device_local
            })
            .ok_or_else(|| {
                SharedMemoryError::Vulkan(
                    "no device-local memory type supports OPAQUE_FD export".into(),
                )
            })?;

        // 4. Allocate memory with export-info chained in.
        let vk_memory = {
            let mut export_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_type_idx)
                .push_next(&mut export_info);
            // SAFETY: alloc_info and chain are valid for the call duration.
            unsafe {
                ash_device.allocate_memory(&alloc_info, None).map_err(|e| {
                    SharedMemoryError::Vulkan(format!("allocate_memory (shared-fft): {e:?}"))
                })?
            }
        };

        // Bind buffer to memory.
        // SAFETY: vk_buffer and vk_memory are both valid and unbound.
        unsafe {
            ash_device
                .bind_buffer_memory(vk_buffer, vk_memory, 0)
                .map_err(|e| {
                    SharedMemoryError::Vulkan(format!("bind_buffer_memory (shared-fft): {e:?}"))
                })?;
        }

        // 5. Export as a Unix fd via VK_KHR_external_memory_fd.
        let raw_fd = {
            let loader = external_memory_fd::Device::new(&ash_instance, &ash_device);
            let fd_info = vk::MemoryGetFdInfoKHR::default()
                .memory(vk_memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            // SAFETY: vk_memory is valid and was allocated with OPAQUE_FD export.
            unsafe {
                loader
                    .get_memory_fd(&fd_info)
                    .map_err(|e| SharedMemoryError::Vulkan(format!("vkGetMemoryFdKHR: {e:?}")))?
            }
        };

        // 6. Import the fd into CUDA. CUDA takes ownership of the fd when
        //    cudaImportExternalMemory succeeds; do not close it ourselves.
        let mut ext_mem_handle: sys::cudaExternalMemory_t = std::ptr::null_mut();
        {
            // SAFETY: cudaExternalMemoryHandleDesc is a C POD struct whose
            // all-zero representation is valid; we overwrite every meaningful
            // field immediately below. The `handle` union field is set via the
            // `fd` variant, which is what OPAQUE_FD requires.
            let mut desc: sys::cudaExternalMemoryHandleDesc = unsafe { std::mem::zeroed() };
            desc.type_ = sys::cudaExternalMemoryHandleType_cudaExternalMemoryHandleTypeOpaqueFd;
            // `handle` is a C union; the `fd` variant is correct for
            // OPAQUE_FD and is the only field CUDA reads.
            desc.handle.fd = raw_fd;
            desc.size = mem_req.size;
            desc.flags = 0;

            // SAFETY: ext_mem_handle is a valid out-pointer; desc is fully
            // initialised; raw_fd is a valid file descriptor; CUDA takes fd
            // ownership on success.
            let rc = unsafe { sys::cudaImportExternalMemory(&mut ext_mem_handle, &desc) };
            if rc != sys::cudaError_cudaSuccess {
                // CUDA didn't take ownership on failure; wrap the fd in an
                // OwnedFd so it is closed when we return the error.
                // SAFETY: raw_fd is a valid open file descriptor that we own.
                let _guard = unsafe { OwnedFd::from_raw_fd(raw_fd) };
                // Best-effort Vulkan cleanup before returning.
                unsafe {
                    ash_device.destroy_buffer(vk_buffer, None);
                    ash_device.free_memory(vk_memory, None);
                }
                return Err(SharedMemoryError::Cuda(format!(
                    "cudaImportExternalMemory: {rc:?}"
                )));
            }
        }

        // 7. Map the full range to get a device pointer CUDA can access.
        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        {
            // SAFETY: all-zero is valid for this POD struct.
            let mut buf_desc: sys::cudaExternalMemoryBufferDesc = unsafe { std::mem::zeroed() };
            buf_desc.offset = 0;
            buf_desc.size = mem_req.size;
            buf_desc.flags = 0;

            // SAFETY: device_ptr is a valid out-pointer; ext_mem_handle was
            // just imported successfully; buf_desc is fully initialised.
            let rc = unsafe {
                sys::cudaExternalMemoryGetMappedBuffer(&mut device_ptr, ext_mem_handle, &buf_desc)
            };
            if rc != sys::cudaError_cudaSuccess {
                // Best-effort cleanup before returning the error.
                unsafe { sys::cudaDestroyExternalMemory(ext_mem_handle) };
                unsafe {
                    ash_device.destroy_buffer(vk_buffer, None);
                    ash_device.free_memory(vk_memory, None);
                }
                return Err(SharedMemoryError::Cuda(format!(
                    "cudaExternalMemoryGetMappedBuffer: {rc:?}"
                )));
            }
        }

        Ok(Self {
            vk_buffer,
            vk_memory,
            ash_device,
            len_complex,
            ext_mem_handle,
            device_ptr,
        })
    }

    /// Raw `VkBuffer` handle for the Vulkan FFT path.
    ///
    /// Valid as long as `self` is alive. Do not destroy it; `Drop` handles
    /// teardown.
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.vk_buffer
    }

    /// Raw CUDA device pointer to the start of the buffer.
    ///
    /// Cast to `*mut cufftComplex` / `*mut f32` when supplying the buffer to
    /// cuFFT exec calls or raw CUDA kernels. Valid as long as `self` is alive.
    /// Do not `cudaFree` it; `Drop` handles teardown.
    pub fn cuda_device_ptr(&self) -> *mut c_void {
        self.device_ptr
    }

    /// Number of `Complex32` elements this buffer holds.
    pub fn len(&self) -> usize {
        self.len_complex
    }

    /// `true` if the buffer holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.len_complex == 0
    }

    /// Upload `host` into the shared device-local memory via
    /// `cudaMemcpy(HostToDevice)`.
    ///
    /// Synchronous: returns only after the copy is complete on the implicit
    /// per-thread CUDA stream 0.
    ///
    /// # Panics
    ///
    /// Panics if `host.len() != self.len()`.
    pub fn upload(&self, host: &[Complex32]) -> Result<(), SharedMemoryError> {
        assert_eq!(
            host.len(),
            self.len_complex,
            "upload: host slice length ({}) must match buffer length ({})",
            host.len(),
            self.len_complex,
        );
        let size = (self.len_complex * core::mem::size_of::<Complex32>()) as usize;
        // SAFETY: device_ptr is a valid CUDA device allocation of `size` bytes;
        // host.as_ptr() is a valid host pointer for `size` bytes; there is no
        // concurrent CUDA access (caller's responsibility per the doc contract);
        // cudaMemcpy with HostToDevice copies from the host pointer to the
        // device pointer.
        let rc = unsafe {
            sys::cudaMemcpy(
                self.device_ptr,
                host.as_ptr().cast::<c_void>(),
                size,
                sys::cudaMemcpyKind_cudaMemcpyHostToDevice,
            )
        };
        if rc != sys::cudaError_cudaSuccess {
            return Err(SharedMemoryError::Cuda(format!(
                "cudaMemcpy(host-to-device): {rc:?}"
            )));
        }
        Ok(())
    }

    /// Download the buffer's contents back to a host `Vec<Complex32>` via
    /// `cudaMemcpy(DeviceToHost)`.
    ///
    /// Synchronous: returns only after the copy is complete on the implicit
    /// per-thread CUDA stream 0.
    pub fn download(&self) -> Result<Vec<Complex32>, SharedMemoryError> {
        let mut out = vec![Complex32::new(0.0, 0.0); self.len_complex];
        let size = (self.len_complex * core::mem::size_of::<Complex32>()) as usize;
        // SAFETY: out.as_mut_ptr() is a valid host pointer for `size` bytes;
        // device_ptr is a valid CUDA device allocation of `size` bytes; there
        // is no concurrent CUDA write (caller's responsibility); cudaMemcpy
        // with DeviceToHost copies from device to the host output buffer.
        let rc = unsafe {
            sys::cudaMemcpy(
                out.as_mut_ptr().cast::<c_void>(),
                self.device_ptr,
                size,
                sys::cudaMemcpyKind_cudaMemcpyDeviceToHost,
            )
        };
        if rc != sys::cudaError_cudaSuccess {
            return Err(SharedMemoryError::Cuda(format!(
                "cudaMemcpy(device-to-host): {rc:?}"
            )));
        }
        Ok(out)
    }
}

impl Drop for SharedFftBuffer {
    fn drop(&mut self) {
        // Tear down in reverse construction order:
        //   1. CUDA mapped buffer pointer (cudaFree)
        //   2. CUDA external-memory handle (cudaDestroyExternalMemory)
        //   3. VkBuffer
        //   4. VkDeviceMemory

        // SAFETY: device_ptr was returned by cudaExternalMemoryGetMappedBuffer
        // and has not been freed elsewhere; double-free is guarded by Drop
        // being called exactly once.
        unsafe {
            sys::cudaFree(self.device_ptr);
        }

        // SAFETY: ext_mem_handle was returned by cudaImportExternalMemory and
        // is still valid; the mapped buffer above must be freed first (done).
        unsafe {
            sys::cudaDestroyExternalMemory(self.ext_mem_handle);
        }

        // SAFETY: vk_buffer and vk_memory are valid handles owned by `self`;
        // ash_device outlives them here because it is stored in `self`.
        unsafe {
            self.ash_device.destroy_buffer(self.vk_buffer, None);
            self.ash_device.free_memory(self.vk_memory, None);
        }
    }
}
