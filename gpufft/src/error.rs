//! Crate-wide error aliases.
//!
//! Each backend defines its own concrete error type; this module re-exports
//! them under a common name for convenience.

#[cfg(feature = "vulkan")]
#[cfg_attr(docsrs, doc(cfg(feature = "vulkan")))]
pub use crate::vulkan::VulkanError;

#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub use crate::cuda::CudaError;

/// Shorthand for the default backend's error type.
///
/// Resolves to [`VulkanError`] when the `vulkan` feature is enabled, or
/// [`CudaError`] when only `cuda` is enabled.
#[cfg(feature = "vulkan")]
pub type Error = VulkanError;

#[cfg(all(feature = "cuda", not(feature = "vulkan")))]
pub type Error = CudaError;
