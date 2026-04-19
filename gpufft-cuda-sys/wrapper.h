/*
 * gpufft-cuda-sys: bindgen surface for cuFFT + the CUDA Runtime subset we
 * need for memory management. Filtered down to the FFT and buffer paths by
 * the allowlists in build.rs.
 */

#ifndef GPUFFT_CUDA_WRAPPER_H
#define GPUFFT_CUDA_WRAPPER_H

#include <cuda_runtime.h>
#include <cufft.h>

#endif /* GPUFFT_CUDA_WRAPPER_H */
