//! Integration tests for the CUDA backend.
//!
//! Skip gracefully if no CUDA device is available so the suite still runs
//! on machines without NVIDIA hardware. cuFFT does not normalise inverses,
//! so tests scale by the element count before comparing.

#![cfg(feature = "cuda")]

use gpufft::{
    BufferOps, C2cPlanOps, C2rPlanOps, Device, Direction, PlanDesc, R2cPlanOps, Shape,
    cuda::{CudaBackend, CudaError, DeviceOptions},
};
use num_complex::{Complex32, Complex64};

fn init_device() -> Option<<CudaBackend as gpufft::Backend>::Device> {
    match CudaBackend::new_device(DeviceOptions::default()) {
        Ok(d) => Some(d),
        Err(CudaError::NoDevice) | Err(CudaError::DeviceOutOfRange { .. }) => None,
        Err(CudaError::Runtime { .. }) => None,
        Err(e) => panic!("device init failed: {e}"),
    }
}

fn scale_complex32(v: &mut [Complex32], s: f32) {
    for x in v {
        x.re *= s;
        x.im *= s;
    }
}

fn scale_f32(v: &mut [f32], s: f32) {
    for x in v {
        *x *= s;
    }
}

#[test]
fn cuda_c2c_1d_complex32() {
    let Some(dev) = init_device() else { return };

    let n = 1024u32;
    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 0.3).cos()))
        .collect();

    let mut buf = dev.alloc::<Complex32>(n as usize).unwrap();
    buf.write(&host).unwrap();

    let mut plan = dev
        .plan_c2c::<Complex32>(&PlanDesc {
            shape: Shape::D1(n),
            batch: 1,
            normalize: false,
        })
        .unwrap();

    plan.execute(&mut buf, Direction::Forward).unwrap();
    plan.execute(&mut buf, Direction::Inverse).unwrap();

    let mut back = vec![Complex32::default(); n as usize];
    buf.read(&mut back).unwrap();
    scale_complex32(&mut back, 1.0 / n as f32);

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-4, "L-inf = {linf} exceeds 1e-4");
}

#[test]
fn cuda_c2c_3d_complex32() {
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let total = (nx * ny * nz) as usize;
    let host: Vec<Complex32> = (0..total)
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 1.1).cos()))
        .collect();

    let mut buf = dev.alloc::<Complex32>(total).unwrap();
    buf.write(&host).unwrap();

    let mut plan = dev
        .plan_c2c::<Complex32>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            batch: 1,
            normalize: false,
        })
        .unwrap();

    plan.execute(&mut buf, Direction::Forward).unwrap();
    plan.execute(&mut buf, Direction::Inverse).unwrap();

    let mut back = vec![Complex32::default(); total];
    buf.read(&mut back).unwrap();
    scale_complex32(&mut back, 1.0 / total as f32);

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-3, "3D L-inf = {linf} exceeds 1e-3");
}

#[test]
fn cuda_r2c_c2r_3d_f32() {
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let real_total = (nx * ny * nz) as usize;
    let complex_total = (nx as u64 * ny as u64 * (nz as u64 / 2 + 1)) as usize;

    let host: Vec<f32> = (0..real_total)
        .map(|i| ((i as f32) * 0.17).sin() + ((i as f32) * 0.53).cos())
        .collect();

    let mut real_in = dev.alloc::<f32>(real_total).unwrap();
    real_in.write(&host).unwrap();
    let mut spectrum = dev.alloc::<Complex32>(complex_total).unwrap();
    let mut real_out = dev.alloc::<f32>(real_total).unwrap();

    let mut r2c = dev
        .plan_r2c::<f32>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            batch: 1,
            normalize: false,
        })
        .unwrap();
    r2c.execute(&real_in, &mut spectrum).unwrap();

    let mut c2r = dev
        .plan_c2r::<f32>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            batch: 1,
            normalize: false,
        })
        .unwrap();
    c2r.execute(&spectrum, &mut real_out).unwrap();

    let mut back = vec![0.0f32; real_total];
    real_out.read(&mut back).unwrap();
    scale_f32(&mut back, 1.0 / real_total as f32);

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-3, "R2C->C2R L-inf = {linf} exceeds 1e-3");
}

#[test]
fn cuda_c2c_3d_complex64() {
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (16u32, 16u32, 16u32);
    let total = (nx * ny * nz) as usize;
    let host: Vec<Complex64> = (0..total)
        .map(|i| Complex64::new((i as f64 * 0.17).sin(), (i as f64 * 0.29).cos()))
        .collect();

    let mut buf = dev.alloc::<Complex64>(total).unwrap();
    buf.write(&host).unwrap();

    let mut plan = dev
        .plan_c2c::<Complex64>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            batch: 1,
            normalize: false,
        })
        .unwrap();

    plan.execute(&mut buf, Direction::Forward).unwrap();
    plan.execute(&mut buf, Direction::Inverse).unwrap();

    let mut back = vec![Complex64::default(); total];
    buf.read(&mut back).unwrap();
    for x in back.iter_mut() {
        x.re /= total as f64;
        x.im /= total as f64;
    }

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(linf < 1e-10, "Complex64 L-inf = {linf} exceeds 1e-10");
}
