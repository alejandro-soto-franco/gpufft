//! Zero-copy memory shared between the Vulkan and CUDA backends.
//!
//! Allocates exportable `VkDeviceMemory` via Vulkan with
//! `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`, retrieves a file
//! descriptor via `vkGetMemoryFdKHR`, and imports it into CUDA through
//! the runtime API (`cudaImportExternalMemory`).
//!
//! On Linux with NVIDIA's proprietary driver, both APIs end up pointing
//! at the same physical bytes, with no host roundtrip and no staging buffer.
//!
//! Same-physical-GPU is assumed: importing memory from a different GPU
//! than CUDA is bound to will either fail outright or silently produce
//! a non-shared mapping. The minimal demo path here doesn't check the
//! adapter↔device UUID; production use should.
//!
//! This module ships the foundation only: a host-visible mirror that
//! both backends can read/write. Promoting it to a full `SharedFftBuffer`
//! that VkFftBackend and CuFftBackend can run FFTs against is tracked
//! separately (Task 5).

use std::ffi::c_void;
use std::os::unix::io::{FromRawFd, OwnedFd};

use ash::khr::external_memory_fd;
use ash::vk;
use gpufft_cuda_sys as sys;

use super::SharedMemoryError;

/// A block of GPU-resident memory addressable from both Vulkan and CUDA.
///
/// On Linux with NVIDIA's proprietary driver, both APIs end up pointing
/// at the same physical bytes, with no host roundtrip and no staging buffer.
pub struct SharedMemory {
    /// The Vulkan buffer handle backed by `vk_memory`.
    vk_buffer: vk::Buffer,
    /// The exportable Vulkan device memory.
    vk_memory: vk::DeviceMemory,
    /// Cloned ash logical device; needed for map/unmap and teardown.
    ash_device: ash::Device,
    /// Allocation size in bytes (may be >= requested due to alignment).
    alloc_size: u64,
    /// Requested size in bytes (used for bounds checking in write_host_bytes).
    size_bytes: u64,
    /// CUDA external-memory handle (owns the fd on the CUDA side).
    ext_mem_handle: sys::cudaExternalMemory_t,
    /// CUDA device pointer into the imported memory (`cudaFree` on drop).
    device_ptr: *mut c_void,
}

// SAFETY: The raw pointers `ext_mem_handle` and `device_ptr` are CUDA/Vulkan
// opaque handles. They are never dereferenced from Rust and only passed back
// to the respective driver APIs. Ownership is with `SharedMemory`; drivers are
// thread-safe for handle destruction.
unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Allocate `size_bytes` of host-visible, host-coherent memory on the
    /// Vulkan device, export it as an `OPAQUE_FD` handle, and import it into
    /// the CUDA context so both backends share the same physical bytes.
    pub fn new(
        vk_dev: &crate::vulkan::VulkanDevice,
        cuda_dev: &crate::cuda::CudaDevice,
        size_bytes: u64,
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

        // 1. Raw ash::Device + ash::Instance from VulkanDevice.
        let handles = vk_dev.raw_handles();
        let ash_device = handles.device.clone();
        let ash_instance = handles.instance.clone();
        let phys_dev = handles.physical_device; // already ash::vk::PhysicalDevice

        // 2. VkBuffer flagged for OPAQUE_FD export.
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
                ash_device
                    .create_buffer(&buf_info, None)
                    .map_err(|e| SharedMemoryError::Vulkan(format!("create_buffer: {e:?}")))?
            }
        };

        // 3. Memory requirements + host-visible-coherent memory type index.
        // SAFETY: vk_buffer was just created above and is valid.
        let mem_req = unsafe { ash_device.get_buffer_memory_requirements(vk_buffer) };

        // SAFETY: phys_dev and ash_instance are valid for the device lifetime.
        let mem_props =
            unsafe { ash_instance.get_physical_device_memory_properties(phys_dev) };

        let wanted =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mem_type_idx = (0..mem_props.memory_type_count)
            .find(|&i| {
                let supported = (mem_req.memory_type_bits & (1 << i)) != 0;
                let has_flags = mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(wanted);
                supported && has_flags
            })
            .ok_or_else(|| {
                SharedMemoryError::Vulkan(
                    "no host-visible coherent memory type supports OPAQUE_FD export".into(),
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
                ash_device
                    .allocate_memory(&alloc_info, None)
                    .map_err(|e| SharedMemoryError::Vulkan(format!("allocate_memory: {e:?}")))?
            }
        };

        // Bind buffer to memory.
        // SAFETY: vk_buffer and vk_memory are both valid and unbound.
        unsafe {
            ash_device
                .bind_buffer_memory(vk_buffer, vk_memory, 0)
                .map_err(|e| SharedMemoryError::Vulkan(format!("bind_buffer_memory: {e:?}")))?;
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
            let mut desc: sys::cudaExternalMemoryHandleDesc =
                unsafe { std::mem::zeroed() };
            desc.type_ =
                sys::cudaExternalMemoryHandleType_cudaExternalMemoryHandleTypeOpaqueFd;
            // `handle` is a C union; the `fd` variant is correct for
            // OPAQUE_FD and is the only field CUDA reads.
            desc.handle.fd = raw_fd;
            desc.size = mem_req.size;
            desc.flags = 0;

            // SAFETY: ext_mem_handle is a valid out-pointer; desc is fully
            // initialised; raw_fd is a valid file descriptor; CUDA takes fd
            // ownership on success.
            let rc = unsafe {
                sys::cudaImportExternalMemory(&mut ext_mem_handle, &desc)
            };
            if rc != sys::cudaError_cudaSuccess {
                // CUDA didn't take ownership on failure; wrap the fd in an
                // OwnedFd so it is closed when we return the error.
                // SAFETY: raw_fd is a valid open file descriptor that we own.
                let _guard = unsafe { OwnedFd::from_raw_fd(raw_fd) };
                return Err(SharedMemoryError::Cuda(format!(
                    "cudaImportExternalMemory: {rc:?}"
                )));
            }
        }

        // 7. Map the full range to get a CUdeviceptr CUDA can access.
        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        {
            // SAFETY: all-zero is valid for this POD struct.
            let mut buf_desc: sys::cudaExternalMemoryBufferDesc =
                unsafe { std::mem::zeroed() };
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
                unsafe { ash_device.destroy_buffer(vk_buffer, None) };
                unsafe { ash_device.free_memory(vk_memory, None) };
                return Err(SharedMemoryError::Cuda(format!(
                    "cudaExternalMemoryGetMappedBuffer: {rc:?}"
                )));
            }
        }

        Ok(Self {
            vk_buffer,
            vk_memory,
            ash_device,
            alloc_size: mem_req.size,
            size_bytes,
            ext_mem_handle,
            device_ptr,
        })
    }

    /// Map the Vulkan host-visible side and copy `bytes` into the shared
    /// allocation.
    ///
    /// Panics if `bytes.len()` exceeds the allocation size. The caller is
    /// responsible for ensuring no concurrent CUDA reads during the copy.
    pub fn write_host_bytes(&self, bytes: &[u8]) -> Result<(), SharedMemoryError> {
        assert!(
            bytes.len() as u64 <= self.size_bytes,
            "write_host_bytes: {} bytes exceeds allocation size {}",
            bytes.len(),
            self.size_bytes,
        );
        // SAFETY: vk_memory is valid and host-visible; we unmap before return.
        unsafe {
            let ptr = self
                .ash_device
                .map_memory(
                    self.vk_memory,
                    0,
                    self.alloc_size,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| SharedMemoryError::Vulkan(format!("map_memory: {e:?}")))?;
            // SAFETY: ptr is a valid host pointer covering at least alloc_size
            // bytes; bytes.len() <= size_bytes <= alloc_size.
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            self.ash_device.unmap_memory(self.vk_memory);
        }
        Ok(())
    }

    /// Return the CUDA device pointer into the shared allocation.
    ///
    /// The pointer is valid as long as `self` is alive. Do not `cudaFree` it;
    /// `Drop` handles cleanup.
    pub fn cuda_device_ptr(&self) -> *mut c_void {
        self.device_ptr
    }

    /// Size of the allocation as requested (may be smaller than the actual
    /// allocation due to alignment rounding).
    pub fn len(&self) -> u64 {
        self.size_bytes
    }

    /// `true` if the requested size was zero.
    pub fn is_empty(&self) -> bool {
        self.size_bytes == 0
    }
}

impl Drop for SharedMemory {
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
