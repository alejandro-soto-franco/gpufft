//! Error type for the Vulkan backend.

use thiserror::Error;

/// Errors returned by the Vulkan backend.
#[derive(Debug, Error)]
pub enum VulkanError {
    /// Failed to load the Vulkan loader (no `libvulkan.so` on the system).
    #[error("failed to load Vulkan loader: {0}")]
    LoaderLoad(#[source] ash::LoadingError),

    /// The requested validation layer is not available.
    #[error("validation layer not available: {0}")]
    ValidationUnavailable(String),

    /// No physical device supports compute.
    #[error("no compute-capable physical device found")]
    NoDevice,

    /// A raw Vulkan API call failed.
    #[error("Vulkan error: {context}: {code:?}")]
    VkResult {
        /// Human-readable phase ("create_instance", "allocate_memory", ...).
        context: &'static str,
        /// The raw `VkResult`.
        code: ash::vk::Result,
    },

    /// No memory type on the physical device satisfied the requested flags.
    #[error("no memory type with required property flags available")]
    NoSuitableMemoryType,

    /// A buffer or slice length did not match the plan's expected extent.
    #[error("length mismatch: expected {expected} elements, got {got}")]
    LengthMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count observed.
        got: usize,
    },

    /// VkFFT returned a nonzero result code.
    #[error("VkFFT error {code}")]
    VkFft {
        /// Raw `VkFFTResult` code.
        code: i32,
    },

    /// A feature is not implemented in the current Vulkan backend build.
    #[error("not implemented in Vulkan backend: {0}")]
    Unsupported(&'static str),

    /// A requested scalar type is not yet supported by VkFFT on this build.
    #[error("scalar type not supported by Vulkan backend: {0}")]
    UnsupportedScalar(&'static str),

    /// Plan specification was internally inconsistent (e.g. batch > 1 in 2D/3D).
    #[error("invalid plan: {0}")]
    InvalidPlan(&'static str),
}

impl VulkanError {
    /// Helper: wrap a `Result<T, ash::vk::Result>` into [`VulkanError::VkResult`].
    pub(crate) fn vk(context: &'static str, code: ash::vk::Result) -> Self {
        Self::VkResult { context, code }
    }
}
