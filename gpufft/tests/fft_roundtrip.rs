//! Integration tests: FFT forward then inverse must recover the input.

#![cfg(feature = "vulkan")]

use gpufft::{
    BufferOps, Device, Direction, PlanDesc, PlanOps, Shape, Transform, vulkan::VulkanBackend,
};
use num_complex::Complex32;

fn init_device() -> Option<<VulkanBackend as gpufft::Backend>::Device> {
    match VulkanBackend::new_device(gpufft::vulkan::DeviceOptions::default()) {
        Ok(d) => Some(d),
        Err(gpufft::vulkan::VulkanError::LoaderLoad(_))
        | Err(gpufft::vulkan::VulkanError::NoDevice) => None,
        Err(e) => panic!("device init failed: {e}"),
    }
}

#[test]
fn roundtrip_1d_complex32() {
    let Some(dev) = init_device() else { return };

    let n = 1024u32;
    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 0.3).cos()))
        .collect();

    let mut buf = dev.alloc::<Complex32>(n as usize).unwrap();
    buf.write(&host).unwrap();

    let mut plan = dev
        .plan::<Complex32>(&PlanDesc {
            shape: Shape::D1(n),
            transform: Transform::C2c,
            batch: 1,
            normalize: true,
        })
        .unwrap();

    plan.execute(&mut buf, Direction::Forward).unwrap();
    plan.execute(&mut buf, Direction::Inverse).unwrap();

    let mut back = vec![Complex32::default(); n as usize];
    buf.read(&mut back).unwrap();

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-4, "L-inf = {linf} exceeds 1e-4");
}

#[test]
fn roundtrip_2d_complex32() {
    let Some(dev) = init_device() else { return };

    let (nx, ny) = (64u32, 64u32);
    let host: Vec<Complex32> = (0..(nx * ny))
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 0.7).cos()))
        .collect();

    let mut buf = dev.alloc::<Complex32>((nx * ny) as usize).unwrap();
    buf.write(&host).unwrap();

    let mut plan = dev
        .plan::<Complex32>(&PlanDesc {
            shape: Shape::D2([nx, ny]),
            transform: Transform::C2c,
            batch: 1,
            normalize: true,
        })
        .unwrap();

    plan.execute(&mut buf, Direction::Forward).unwrap();
    plan.execute(&mut buf, Direction::Inverse).unwrap();

    let mut back = vec![Complex32::default(); (nx * ny) as usize];
    buf.read(&mut back).unwrap();

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-4, "2D L-inf = {linf}");
}

#[test]
fn roundtrip_3d_complex32() {
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let total = (nx * ny * nz) as usize;
    let host: Vec<Complex32> = (0..total)
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 1.1).cos()))
        .collect();

    let mut buf = dev.alloc::<Complex32>(total).unwrap();
    buf.write(&host).unwrap();

    let mut plan = dev
        .plan::<Complex32>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            transform: Transform::C2c,
            batch: 1,
            normalize: true,
        })
        .unwrap();

    plan.execute(&mut buf, Direction::Forward).unwrap();
    plan.execute(&mut buf, Direction::Inverse).unwrap();

    let mut back = vec![Complex32::default(); total];
    buf.read(&mut back).unwrap();

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-4, "3D L-inf = {linf}");
}
