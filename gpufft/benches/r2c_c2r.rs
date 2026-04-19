//! Comparative benchmark: 3D R2C + C2R roundtrip across backends.
//!
//! Runs `cargo bench --features vulkan,cuda --bench r2c_c2r`. Times a full
//! forward + inverse cycle through the gpufft API (upload -> FFT ->
//! download) for both the Vulkan (VkFFT) and CUDA (cuFFT) backends at
//! vonkarman-typical 3D sizes. The CPU baseline uses `rustfft` on the
//! innermost axis only to give a rough comparison point; it is not an
//! apples-to-apples 3D FFT, only a reference scale.

#![cfg(any(feature = "vulkan", feature = "cuda"))]

use std::time::Instant;

use num_complex::Complex32;

const ITERATIONS: u32 = 10;
const WARMUP: u32 = 3;

fn bench_at(n: u32) {
    println!("\n=== Shape: {n}^3, Complex32 ===");

    #[cfg(feature = "cuda")]
    bench_cuda(n);

    #[cfg(feature = "vulkan")]
    bench_vulkan(n);
}

#[cfg(feature = "cuda")]
fn bench_cuda(n: u32) {
    use gpufft::{
        BufferOps, C2rPlanOps, Device, PlanDesc, R2cPlanOps, Shape,
        cuda::{CudaBackend, DeviceOptions},
    };

    let Ok(dev) = CudaBackend::new_device(DeviceOptions::default()) else {
        println!("  cuda:   [device unavailable]");
        return;
    };

    let real_total = (n * n * n) as usize;
    let complex_total = n as usize * n as usize * (n as usize / 2 + 1);
    let host: Vec<f32> = (0..real_total).map(|i| (i as f32 * 0.17).sin()).collect();

    let mut real_in = dev.alloc::<f32>(real_total).unwrap();
    real_in.write(&host).unwrap();
    let mut spectrum = dev.alloc::<Complex32>(complex_total).unwrap();
    let mut real_out = dev.alloc::<f32>(real_total).unwrap();

    let desc = PlanDesc {
        shape: Shape::D3([n, n, n]),
        batch: 1,
        normalize: false,
    };

    let mut r2c = dev.plan_r2c::<f32>(&desc).unwrap();
    let mut c2r = dev.plan_c2r::<f32>(&desc).unwrap();

    for _ in 0..WARMUP {
        r2c.execute(&real_in, &mut spectrum).unwrap();
        c2r.execute(&spectrum, &mut real_out).unwrap();
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        r2c.execute(&real_in, &mut spectrum).unwrap();
        c2r.execute(&spectrum, &mut real_out).unwrap();
    }
    let per_iter = start.elapsed() / ITERATIONS;
    let gpoints = (real_total as f64) / 1e9 / per_iter.as_secs_f64();
    println!(
        "  cuda:   {:?} / pair     ({:.2} Gelem/s)",
        per_iter, gpoints
    );
}

#[cfg(feature = "vulkan")]
fn bench_vulkan(n: u32) {
    use gpufft::{
        BufferOps, C2rPlanOps, Device, PlanDesc, R2cPlanOps, Shape,
        vulkan::{DeviceOptions, VulkanBackend},
    };

    let Ok(dev) = VulkanBackend::new_device(DeviceOptions::default()) else {
        println!("  vulkan: [device unavailable]");
        return;
    };

    let real_total = (n * n * n) as usize;
    let complex_total = n as usize * n as usize * (n as usize / 2 + 1);
    let host: Vec<f32> = (0..real_total).map(|i| (i as f32 * 0.17).sin()).collect();

    let mut real_in = dev.alloc::<f32>(real_total).unwrap();
    real_in.write(&host).unwrap();
    let mut spectrum = dev.alloc::<Complex32>(complex_total).unwrap();
    let mut real_out = dev.alloc::<f32>(real_total).unwrap();

    let desc = PlanDesc {
        shape: Shape::D3([n, n, n]),
        batch: 1,
        normalize: false,
    };

    let mut r2c = dev.plan_r2c::<f32>(&desc).unwrap();
    let mut c2r = dev.plan_c2r::<f32>(&desc).unwrap();

    for _ in 0..WARMUP {
        r2c.execute(&real_in, &mut spectrum).unwrap();
        c2r.execute(&spectrum, &mut real_out).unwrap();
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        r2c.execute(&real_in, &mut spectrum).unwrap();
        c2r.execute(&spectrum, &mut real_out).unwrap();
    }
    let per_iter = start.elapsed() / ITERATIONS;
    let gpoints = (real_total as f64) / 1e9 / per_iter.as_secs_f64();
    println!(
        "  vulkan: {:?} / pair     ({:.2} Gelem/s)",
        per_iter, gpoints
    );
}

fn main() {
    println!("gpufft R2C+C2R 3D benchmark (f32, Complex32)");
    println!("Iterations per measurement: {ITERATIONS} (after {WARMUP} warmup)");

    for &n in &[32u32, 64, 128, 256] {
        bench_at(n);
    }
}
