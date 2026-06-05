//! End-to-end zero-copy roundtrip across Vulkan and CUDA.
//!
//! Allocates a [`gpufft::shared::SharedFftBuffer`], writes a known host
//! pattern into it, runs a Vulkan forward FFT in place via
//! [`gpufft::vulkan::VulkanC2cPlan::execute_shared`], then runs a CUDA
//! inverse on the exact same allocation via
//! [`gpufft::cuda::CudaC2cPlan::execute_shared`]. Because neither backend
//! normalises, the result equals N * input; we divide on the host and
//! compare.
//!
//! Skipped on platforms or systems missing either backend.

#![cfg(all(feature = "shared", target_os = "linux"))]

use gpufft::{
    Device, Direction, PlanDesc, Shape, cuda::CudaBackend, shared::SharedFftBuffer,
    vulkan::VulkanBackend,
};
use num_complex::Complex32;

const N: usize = 256;
// Absolute tolerance is on the recovered (scale-divided) values. Per-element
// FP32 drift accumulates with magnitude; at i=255 the input is 255.0+127.5i
// and observed delta on RTX 5060 + cuFFT 13.1 + VkFFT is ~1.2e-4. A 5e-4
// absolute bound is tight enough to catch real bugs (off-by-one normalisation
// would be 256x or 1/256x = O(1) error) without flaking on FP32.
const TOL: f32 = 5e-4;

#[test]
fn vulkan_forward_cuda_inverse_roundtrip() {
    let vk_dev = match VulkanBackend::new_device(Default::default()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: Vulkan device unavailable: {e}");
            return;
        }
    };
    let cu_dev = match CudaBackend::new_device(Default::default()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip: CUDA device unavailable: {e}");
            return;
        }
    };

    let buf = match SharedFftBuffer::new(&vk_dev, &cu_dev, N) {
        Ok(b) => b,
        Err(e) => {
            // External-memory FD path may not be available (e.g., non-NVIDIA
            // Vulkan driver, missing VK_KHR_external_memory_fd, or
            // cross-GPU import). That is not a test failure -- skip.
            eprintln!("skip: SharedFftBuffer unavailable: {e}");
            return;
        }
    };

    // Deterministic input.
    let host_in: Vec<Complex32> = (0..N)
        .map(|i| Complex32::new(i as f32, (i as f32) * 0.5))
        .collect();
    buf.upload(&host_in).expect("upload");

    // Plans (both unnormalized -- see test docstring).
    let mut vk_plan = vk_dev
        .plan_c2c::<Complex32>(&PlanDesc {
            shape: Shape::D1(N as u32),
            batch: 1,
            normalize: false,
        })
        .expect("Vulkan plan_c2c");
    let mut cu_plan = cu_dev
        .plan_c2c::<Complex32>(&PlanDesc {
            shape: Shape::D1(N as u32),
            batch: 1,
            normalize: false,
        })
        .expect("CUDA plan_c2c");

    // Forward on Vulkan, inverse on CUDA -- both against the shared buffer.
    vk_plan
        .execute_shared(&buf, Direction::Forward)
        .expect("Vulkan execute_shared forward");
    cu_plan
        .execute_shared(&buf, Direction::Inverse)
        .expect("CUDA execute_shared inverse");

    let result = buf.download().expect("download");

    // Both backends are unnormalised; inverse(forward(x)) = N * x.
    let scale = N as f32;
    for i in 0..N {
        let expected = host_in[i];
        let got = result[i] / scale;
        let dre = (got.re - expected.re).abs();
        let dim = (got.im - expected.im).abs();
        assert!(
            dre < TOL && dim < TOL,
            "element {i}: expected {expected:?}, got {got:?} (delta re={dre:.3e} im={dim:.3e})"
        );
    }
}
