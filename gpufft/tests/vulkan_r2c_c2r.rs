//! Vulkan R2C / C2R integration tests.

#![cfg(feature = "vulkan")]

use gpufft::{
    BufferOps, C2rPlanOps, Device, PlanDesc, R2cPlanOps, Shape,
    vulkan::{DeviceOptions, VulkanBackend, VulkanError},
};
use num_complex::Complex32;

fn init_device() -> Option<<VulkanBackend as gpufft::Backend>::Device> {
    match VulkanBackend::new_device(DeviceOptions::default()) {
        Ok(d) => Some(d),
        Err(VulkanError::LoaderLoad(_)) | Err(VulkanError::NoDevice) => None,
        Err(e) => panic!("device init failed: {e}"),
    }
}

fn scale_f32(v: &mut [f32], s: f32) {
    for x in v {
        *x *= s;
    }
}

#[test]
fn vulkan_r2c_c2r_1d_f32() {
    let Some(dev) = init_device() else { return };

    let n = 1024u32;
    let host: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.23).sin() + (i as f32 * 0.71).cos())
        .collect();

    let mut real_in = dev.alloc::<f32>(n as usize).unwrap();
    real_in.write(&host).unwrap();

    let complex_len = (n / 2 + 1) as usize;
    let mut spectrum = dev.alloc::<Complex32>(complex_len).unwrap();
    let mut real_out = dev.alloc::<f32>(n as usize).unwrap();

    let desc = PlanDesc {
        shape: Shape::D1(n),
        batch: 1,
        normalize: true,
    };

    let mut r2c = dev.plan_r2c::<f32>(&desc).unwrap();
    r2c.execute(&real_in, &mut spectrum).unwrap();

    let mut c2r = dev.plan_c2r::<f32>(&desc).unwrap();
    c2r.execute(&spectrum, &mut real_out).unwrap();

    let mut back = vec![0.0f32; n as usize];
    real_out.read(&mut back).unwrap();

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-4, "1D R2C->C2R L-inf = {linf} exceeds 1e-4");
}

#[test]
fn vulkan_r2c_c2r_3d_cubic_f32() {
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let real_total = (nx * ny * nz) as usize;
    let complex_total = (nx as u64 * ny as u64 * (nz as u64 / 2 + 1)) as usize;

    let host: Vec<f32> = (0..real_total).map(|i| ((i as f32) * 0.17).sin()).collect();

    let mut real_in = dev.alloc::<f32>(real_total).unwrap();
    real_in.write(&host).unwrap();
    let mut spectrum = dev.alloc::<Complex32>(complex_total).unwrap();
    let mut real_out = dev.alloc::<f32>(real_total).unwrap();

    let desc = PlanDesc {
        shape: Shape::D3([nx, ny, nz]),
        batch: 1,
        normalize: false,
    };

    let mut r2c = dev.plan_r2c::<f32>(&desc).unwrap();
    r2c.execute(&real_in, &mut spectrum).unwrap();

    let mut c2r = dev.plan_c2r::<f32>(&desc).unwrap();
    c2r.execute(&spectrum, &mut real_out).unwrap();

    let mut back = vec![0.0f32; real_total];
    real_out.read(&mut back).unwrap();
    scale_f32(&mut back, 1.0 / real_total as f32);

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-3, "cubic 3D L-inf = {linf}");
}

#[test]
fn vulkan_r2c_c2r_3d_f32() {
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (16u32, 32u32, 64u32);
    let real_total = (nx * ny * nz) as usize;
    let complex_total = (nx as u64 * ny as u64 * (nz as u64 / 2 + 1)) as usize;

    // Non-cubic shape so that any transposed dim-ordering bug would surface
    // as a length or offset mismatch.
    let host: Vec<f32> = (0..real_total)
        .map(|i| ((i as f32) * 0.17).sin() + ((i as f32) * 0.53).cos())
        .collect();

    let mut real_in = dev.alloc::<f32>(real_total).unwrap();
    real_in.write(&host).unwrap();
    let mut spectrum = dev.alloc::<Complex32>(complex_total).unwrap();
    let mut real_out = dev.alloc::<f32>(real_total).unwrap();

    let desc = PlanDesc {
        shape: Shape::D3([nx, ny, nz]),
        batch: 1,
        normalize: false,
    };

    let mut r2c = dev.plan_r2c::<f32>(&desc).unwrap();
    r2c.execute(&real_in, &mut spectrum).unwrap();

    let mut c2r = dev.plan_c2r::<f32>(&desc).unwrap();
    c2r.execute(&spectrum, &mut real_out).unwrap();

    let mut back = vec![0.0f32; real_total];
    real_out.read(&mut back).unwrap();
    scale_f32(&mut back, 1.0 / real_total as f32);

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-3, "3D R2C->C2R L-inf = {linf} exceeds 1e-3");
}

#[test]
fn vulkan_r2c_output_shape_matches_cufft_convention() {
    // For Shape::D3([nx, ny, nz]) the complex half-spectrum must be
    // sized (nx, ny, nz/2+1). This asserts the buffer length path.
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (8u32, 12u32, 16u32);
    let expected_complex = nx as usize * ny as usize * (nz as usize / 2 + 1);

    let host: Vec<f32> = (0..(nx * ny * nz)).map(|i| i as f32).collect();
    let mut real_in = dev.alloc::<f32>(host.len()).unwrap();
    real_in.write(&host).unwrap();

    let mut spectrum = dev.alloc::<Complex32>(expected_complex).unwrap();

    let mut r2c = dev
        .plan_r2c::<f32>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            batch: 1,
            normalize: false,
        })
        .unwrap();

    r2c.execute(&real_in, &mut spectrum)
        .expect("R2C execute should accept (nx, ny, nz/2+1) complex output");
}
