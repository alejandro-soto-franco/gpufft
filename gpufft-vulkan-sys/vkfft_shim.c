/*
 * gpufft-vulkan-sys: extern-C shim bodies.
 *
 * One-line pass-throughs to VkFFT's `static inline` entry points. The
 * compiled object file materializes real symbols that bindgen can bind
 * to. See wrapper.h for contract docs.
 */

#include "wrapper.h"

int gpufft_vkfft_version(void) {
    return VkFFTGetVersion();
}

int gpufft_vkfft_init(VkFFTApplication* app, VkFFTConfiguration cfg) {
    return (int)initializeVkFFT(app, cfg);
}

int gpufft_vkfft_append(VkFFTApplication* app, int inverse, VkFFTLaunchParams* params) {
    return (int)VkFFTAppend(app, inverse, params);
}

void gpufft_vkfft_delete(VkFFTApplication* app) {
    deleteVkFFT(app);
}
