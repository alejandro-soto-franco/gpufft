//! Regression test: a single normalized C2C plan must produce correct
//! round-trips across multiple distinct buffers.
//!
//! cartan-homog's Moulinec-Suquet spectral homogenisation FFTs many
//! per-direction polarisation buffers (`σ_x`, `σ_y`, `σ_z`, etc.) through
//! one cached `VulkanC2cPlan` with `normalize: true`. Before the plan-reuse
//! fix, the second and later calls applied an incorrect normalization
//! factor; the workaround was building a fresh plan per buffer. This test
//! verifies the contract directly.

#![cfg(feature = "vulkan")]

use std::sync::{Mutex, MutexGuard, OnceLock};

use gpufft::{BufferOps, C2cPlanOps, Device, Direction, PlanDesc, Shape, vulkan::VulkanBackend};
use num_complex::Complex32;

fn init_device() -> Option<<VulkanBackend as gpufft::Backend>::Device> {
    match VulkanBackend::new_device(gpufft::vulkan::DeviceOptions::default()) {
        Ok(d) => Some(d),
        Err(gpufft::vulkan::VulkanError::LoaderLoad(_))
        | Err(gpufft::vulkan::VulkanError::NoDevice) => None,
        Err(e) => panic!("device init failed: {e}"),
    }
}

/// Serialize all three tests against a process-wide mutex. Three Vulkan
/// contexts running in parallel on a single GPU exhaust device queue
/// resources and trip `ERROR_DEVICE_LOST` on this host; the bug under
/// test is single-threaded by nature, so contention is unnecessary.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// Reuse one normalized 3D C2C plan across three different buffers, doing a
/// full forward+inverse round-trip on each. With the bug, the second and
/// third buffers' inverse pass produced wrong amplitudes.
#[test]
fn one_plan_three_buffers_roundtrip_3d() {
    let _g = serial();
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let total = (nx * ny * nz) as usize;

    let mut plan = dev
        .plan_c2c::<Complex32>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            batch: 1,
            normalize: true,
        })
        .unwrap();

    // Three distinct input fields, each will get its own GPU buffer.
    let inputs: [Vec<Complex32>; 3] = [
        (0..total)
            .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 0.3).cos()))
            .collect(),
        (0..total)
            .map(|i| Complex32::new((i as f32 * 0.7).cos(), (i as f32).sin()))
            .collect(),
        (0..total)
            .map(|i| Complex32::new(((i % 17) as f32) - 8.0, ((i % 11) as f32) - 5.0))
            .collect(),
    ];

    let mut buffers: Vec<_> = inputs
        .iter()
        .map(|host| {
            let mut buf = dev.alloc::<Complex32>(total).unwrap();
            buf.write(host).unwrap();
            buf
        })
        .collect();

    // Sequential same-direction calls across distinct buffers: the cartan
    // pattern: forward σ_x, forward σ_y, forward σ_z.
    for buf in buffers.iter_mut() {
        plan.execute(buf, Direction::Forward).unwrap();
    }
    // Then inverse on each, so the full round-trip must recover the input.
    for buf in buffers.iter_mut() {
        plan.execute(buf, Direction::Inverse).unwrap();
    }

    for (idx, (host, buf)) in inputs.iter().zip(buffers.iter()).enumerate() {
        let mut back = vec![Complex32::default(); total];
        buf.read(&mut back).unwrap();
        let linf = host
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f32, f32::max);
        assert!(linf < 1e-3, "buffer {idx}: L-inf = {linf} exceeds 1e-3");
    }
}

/// Tighter variant that interleaves forward+inverse per buffer: the
/// pattern a naive caller would use if they didn't know about the bug.
#[test]
fn one_plan_interleaved_roundtrips_3d() {
    let _g = serial();
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let total = (nx * ny * nz) as usize;

    let mut plan = dev
        .plan_c2c::<Complex32>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            batch: 1,
            normalize: true,
        })
        .unwrap();

    for trial in 0..4 {
        let host: Vec<Complex32> = (0..total)
            .map(|i| {
                let f = (i as f32 + trial as f32 * 100.0).to_radians();
                Complex32::new(f.sin(), f.cos())
            })
            .collect();
        let mut buf = dev.alloc::<Complex32>(total).unwrap();
        buf.write(&host).unwrap();
        plan.execute(&mut buf, Direction::Forward).unwrap();
        plan.execute(&mut buf, Direction::Inverse).unwrap();
        let mut back = vec![Complex32::default(); total];
        buf.read(&mut back).unwrap();
        let linf = host
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f32, f32::max);
        assert!(linf < 1e-3, "trial {trial}: L-inf = {linf} exceeds 1e-3");
    }
}

/// Direct numeric assertion on the reporter's exact phrasing: a forward FFT
/// of a constant-1 field should produce DC = N (not N normalized, not
/// 1/N). Test on the SECOND buffer routed through a single plan: that's
/// the call the bug corrupted.
#[test]
fn forward_dc_unchanged_across_buffers_3d() {
    let _g = serial();
    let Some(dev) = init_device() else { return };

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let total = (nx * ny * nz) as usize;
    let n_total_f = total as f32;

    let mut plan = dev
        .plan_c2c::<Complex32>(&PlanDesc {
            shape: Shape::D3([nx, ny, nz]),
            batch: 1,
            normalize: true,
        })
        .unwrap();

    // First call: warm the plan on some buffer.
    let warm_host: Vec<Complex32> = vec![Complex32::new(1.0, 0.0); total];
    let mut warm = dev.alloc::<Complex32>(total).unwrap();
    warm.write(&warm_host).unwrap();
    plan.execute(&mut warm, Direction::Forward).unwrap();
    let mut warm_out = vec![Complex32::default(); total];
    warm.read(&mut warm_out).unwrap();
    let dc_warm = warm_out[0].re;
    assert!(
        (dc_warm - n_total_f).abs() < 1e-2,
        "first-call DC = {dc_warm}, expected {n_total_f}"
    );

    // Second call on a DIFFERENT buffer. DC should still be N.
    let host: Vec<Complex32> = vec![Complex32::new(1.0, 0.0); total];
    let mut buf = dev.alloc::<Complex32>(total).unwrap();
    buf.write(&host).unwrap();
    plan.execute(&mut buf, Direction::Forward).unwrap();
    let mut out = vec![Complex32::default(); total];
    buf.read(&mut out).unwrap();
    let dc = out[0].re;
    assert!(
        (dc - n_total_f).abs() < 1e-2,
        "second-call DC = {dc}, expected {n_total_f} (bug: would be ~1.0)"
    );
}
