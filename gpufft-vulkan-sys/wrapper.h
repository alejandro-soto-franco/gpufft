/*
 * gpufft-vulkan-sys: C wrappers around VkFFT for Rust FFI.
 *
 * VkFFT's public surface uses a pointer-to-handle ABI for Vulkan objects and
 * `static inline` functions, both of which are hostile to direct bindgen
 * consumption. This header exposes a minimal extern-C facade:
 *
 *   - `gpufft_vkfft_init`   wraps `initializeVkFFT`.
 *   - `gpufft_vkfft_append` wraps `VkFFTAppend`.
 *   - `gpufft_vkfft_delete` wraps `deleteVkFFT`.
 *   - `gpufft_vkfft_version` returns the VkFFT runtime version.
 *
 * The Rust side constructs `VkFFTConfiguration` and `VkFFTLaunchParams`
 * directly (bindgen emits both as plain structs). The only reason this
 * header exists is to produce real (non-inline) symbols for the init/append/
 * delete/version entry points.
 */

#ifndef GPUFFT_VKFFT_WRAPPER_H
#define GPUFFT_VKFFT_WRAPPER_H

#define VKFFT_BACKEND 0  /* 0 = Vulkan; see vkFFT.h. */

#include "vkFFT/vkFFT.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Returns VkFFT's runtime version as a packed integer. */
int gpufft_vkfft_version(void);

/**
 * Initialize a VkFFT application from a pre-built configuration.
 *
 * The caller owns the `VkFFTApplication*` (typically zeroed stack or heap
 * memory). All Vulkan handle-pointer fields inside `cfg` (physicalDevice,
 * device, queue, commandPool, fence, buffer, bufferSize) must point to
 * storage that outlives the returned application (VkFFT may retain these
 * pointers for later `VkFFTAppend` calls).
 *
 * Returns 0 (`VKFFT_SUCCESS`) on success, or a nonzero `VkFFTResult` code.
 */
int gpufft_vkfft_init(VkFFTApplication* app, VkFFTConfiguration cfg);

/**
 * Record an FFT dispatch into the caller-provided command buffer.
 *
 * `params->commandBuffer` must point to an allocated `VkCommandBuffer`
 * in the recording state. The dispatch is only recorded; the caller is
 * responsible for ending, submitting, and waiting on the command buffer.
 *
 * Returns 0 (`VKFFT_SUCCESS`) on success, or a nonzero `VkFFTResult` code.
 */
int gpufft_vkfft_append(VkFFTApplication* app, int inverse, VkFFTLaunchParams* params);

/** Release all internal resources owned by the VkFFT application. */
void gpufft_vkfft_delete(VkFFTApplication* app);

#ifdef __cplusplus
}
#endif

#endif /* GPUFFT_VKFFT_WRAPPER_H */
