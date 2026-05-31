//! Python bindings for `gpufft`: one cross-vendor FFT API over cuFFT and VkFFT.
//!
//! The Python surface mirrors `ferrum_gpu` so the two packages are swappable:
//! `gpufft.fft.fft_1d_c2c_pow2(arr, log_n, direction, normalize, device, backend)`.
//! Unlike `ferrum_gpu`, the same call runs on NVIDIA (cuFFT) or on any Vulkan
//! device (VkFFT: AMD, Intel, Apple, NVIDIA) by passing `backend=` or a
//! backend-specific `Device`, and accepts arbitrary (non-power-of-two) sizes
//! via `fft_1d_c2c`.

#![warn(missing_docs)]

use std::sync::Arc;

use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use gpufft::cuda::{CudaBackend, CudaDevice, DeviceOptions as CudaDeviceOptions};
use gpufft::vulkan::{DeviceOptions as VulkanDeviceOptions, VulkanBackend, VulkanDevice};
use gpufft::{Backend, BufferOps, C2cPlanOps, Device as _, Direction, PlanDesc, Shape};

/// In-place C2C transform on a concrete backend device; returns the result.
///
/// `normalize` is always passed to the plan as `false` (cuFFT rejects native
/// normalisation); the Python layer scales inverse output host-side so the
/// behaviour is identical across backends.
fn run_c2c<B: Backend>(
    dev: &B::Device,
    data: &[Complex32],
    shape: Shape,
    batch: u32,
    dir: Direction,
) -> Result<Vec<Complex32>, B::Error> {
    let mut buf = dev.alloc::<Complex32>(data.len())?;
    buf.write(data)?;
    let mut plan = dev.plan_c2c::<Complex32>(&PlanDesc {
        shape,
        batch,
        normalize: false,
    })?;
    plan.execute(&mut buf, dir)?;
    dev.synchronize()?;
    let mut out = vec![Complex32::default(); data.len()];
    buf.read(&mut out)?;
    Ok(out)
}

/// A resolved, Send-able device handle for either backend.
enum DevHandle {
    Cuda(Arc<CudaDevice>),
    Vulkan(Arc<VulkanDevice>),
}

/// Persistent CUDA (cuFFT) device handle. Reuse across calls to amortise
/// device + plan-cache setup.
#[pyclass(module = "gpufft.cuda", name = "Device")]
pub struct PyCudaDevice {
    inner: Arc<CudaDevice>,
}

#[pymethods]
impl PyCudaDevice {
    #[new]
    #[pyo3(signature = (ordinal = 0))]
    fn new(ordinal: i32) -> PyResult<Self> {
        let dev = CudaBackend::new_device(CudaDeviceOptions {
            device_ordinal: Some(ordinal),
        })
        .map_err(|e| PyValueError::new_err(format!("cuda device {ordinal}: {e}")))?;
        Ok(Self {
            inner: Arc::new(dev),
        })
    }

    /// Block until all in-flight GPU work completes.
    fn sync(&self) -> PyResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| PyValueError::new_err(format!("sync: {e}")))
    }

    /// Backend name (`"cuda"`).
    fn backend(&self) -> &'static str {
        "cuda"
    }
}

/// Persistent Vulkan (VkFFT) device handle, usable on any Vulkan device
/// (AMD, Intel, Apple via MoltenVK, NVIDIA).
#[pyclass(module = "gpufft.vulkan", name = "Device")]
pub struct PyVulkanDevice {
    inner: Arc<VulkanDevice>,
}

#[pymethods]
impl PyVulkanDevice {
    #[new]
    #[pyo3(signature = (ordinal = 0))]
    fn new(ordinal: i32) -> PyResult<Self> {
        let _ = ordinal; // Vulkan device selection is a follow-up; uses the default adapter.
        let dev = VulkanBackend::new_device(VulkanDeviceOptions::default())
            .map_err(|e| PyValueError::new_err(format!("vulkan device: {e}")))?;
        Ok(Self {
            inner: Arc::new(dev),
        })
    }

    /// Block until all in-flight GPU work completes.
    fn sync(&self) -> PyResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| PyValueError::new_err(format!("sync: {e}")))
    }

    /// Backend name (`"vulkan"`).
    fn backend(&self) -> &'static str {
        "vulkan"
    }
}

/// Resolve an explicit `device` (either backend's `Device`) or, failing that,
/// build a transient device for the named `backend`.
fn resolve_device(device: Option<&Bound<'_, PyAny>>, backend: &str) -> PyResult<DevHandle> {
    if let Some(obj) = device {
        if let Ok(d) = obj.downcast::<PyCudaDevice>() {
            return Ok(DevHandle::Cuda(d.borrow().inner.clone()));
        }
        if let Ok(d) = obj.downcast::<PyVulkanDevice>() {
            return Ok(DevHandle::Vulkan(d.borrow().inner.clone()));
        }
        return Err(PyValueError::new_err(
            "device must be a gpufft.cuda.Device or gpufft.vulkan.Device",
        ));
    }
    match backend {
        "cuda" => {
            let d = CudaBackend::new_device(CudaDeviceOptions {
                device_ordinal: Some(0),
            })
            .map_err(|e| PyValueError::new_err(format!("cuda device: {e}")))?;
            Ok(DevHandle::Cuda(Arc::new(d)))
        }
        "vulkan" => {
            let d = VulkanBackend::new_device(VulkanDeviceOptions::default())
                .map_err(|e| PyValueError::new_err(format!("vulkan device: {e}")))?;
            Ok(DevHandle::Vulkan(Arc::new(d)))
        }
        other => Err(PyValueError::new_err(format!(
            "backend must be 'cuda' or 'vulkan', got {other:?}"
        ))),
    }
}

fn parse_direction(direction: &str) -> PyResult<Direction> {
    match direction {
        "forward" => Ok(Direction::Forward),
        "inverse" => Ok(Direction::Inverse),
        other => Err(PyValueError::new_err(format!(
            "direction must be 'forward' or 'inverse', got {other:?}"
        ))),
    }
}

/// Core 1D C2C runner shared by the pow2 and arbitrary-N entry points.
fn c2c_1d(
    py: Python<'_>,
    data: &[Complex32],
    n: u32,
    batch: u32,
    dir: Direction,
    normalize: bool,
    handle: DevHandle,
) -> PyResult<Vec<Complex32>> {
    let input = data.to_vec();
    let nn = n as usize;
    py.allow_threads(move || -> Result<Vec<Complex32>, String> {
        let mut out = match handle {
            DevHandle::Cuda(d) => {
                run_c2c::<CudaBackend>(d.as_ref(), &input, Shape::D1(n), batch, dir)
                    .map_err(|e| e.to_string())?
            }
            DevHandle::Vulkan(d) => {
                run_c2c::<VulkanBackend>(d.as_ref(), &input, Shape::D1(n), batch, dir)
                    .map_err(|e| e.to_string())?
            }
        };
        if normalize && matches!(dir, Direction::Inverse) {
            let s = 1.0f32 / nn as f32;
            for v in out.iter_mut() {
                *v *= s;
            }
        }
        Ok(out)
    })
    .map_err(|e| PyValueError::new_err(format!("gpufft fft error: {e}")))
}

/// FFT of a 1D `complex64` array of length `N = 1 << log_n` (batched).
///
/// Drop-in for `ferrum_gpu.fft.fft_1d_c2c_pow2`, plus `backend=` ("cuda" or
/// "vulkan") and an optional persistent `device`.
#[pyfunction]
#[pyo3(signature = (arr, log_n, direction = "forward", normalize = false, device = None, backend = "cuda"))]
fn fft_1d_c2c_pow2<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, Complex32>,
    log_n: u32,
    direction: &str,
    normalize: bool,
    device: Option<&Bound<'py, PyAny>>,
    backend: &str,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    if !(1..=27).contains(&log_n) {
        return Err(PyValueError::new_err(format!(
            "log_n must be in [1, 27], got {log_n}"
        )));
    }
    let n = 1u32 << log_n;
    let data = arr.as_slice()?;
    let total = data.len();
    if total == 0 || total % (n as usize) != 0 {
        return Err(PyValueError::new_err(format!(
            "arr len {total} must be a positive multiple of N = {n}"
        )));
    }
    let batch = (total / n as usize) as u32;
    let dir = parse_direction(direction)?;
    let handle = resolve_device(device, backend)?;
    let out = c2c_1d(py, data, n, batch, dir, normalize, handle)?;
    Ok(out.into_pyarray(py))
}

/// FFT of a 1D `complex64` array of arbitrary length `n` (batched). gpufft's
/// edge over `ferrum_gpu`: non-power-of-two sizes via cuFFT / VkFFT.
#[pyfunction]
#[pyo3(signature = (arr, n, batch = 1, direction = "forward", normalize = false, device = None, backend = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn fft_1d_c2c<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, Complex32>,
    n: u32,
    batch: u32,
    direction: &str,
    normalize: bool,
    device: Option<&Bound<'py, PyAny>>,
    backend: &str,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    if n == 0 || batch == 0 {
        return Err(PyValueError::new_err("n and batch must be positive"));
    }
    let data = arr.as_slice()?;
    let expected = (n as usize) * (batch as usize);
    if data.len() != expected {
        return Err(PyValueError::new_err(format!(
            "arr len {} != n*batch = {expected}",
            data.len()
        )));
    }
    let dir = parse_direction(direction)?;
    let handle = resolve_device(device, backend)?;
    let out = c2c_1d(py, data, n, batch, dir, normalize, handle)?;
    Ok(out.into_pyarray(py))
}

/// Returns the crate version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Backends compiled into this wheel.
#[pyfunction]
fn available_backends() -> Vec<&'static str> {
    vec!["cuda", "vulkan"]
}

/// Native extension module, exposed as `gpufft._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;

    let cuda = PyModule::new(m.py(), "cuda")?;
    cuda.add_class::<PyCudaDevice>()?;
    m.add_submodule(&cuda)?;

    let vulkan = PyModule::new(m.py(), "vulkan")?;
    vulkan.add_class::<PyVulkanDevice>()?;
    m.add_submodule(&vulkan)?;

    let fft = PyModule::new(m.py(), "fft")?;
    fft.add_function(wrap_pyfunction!(fft_1d_c2c_pow2, &fft)?)?;
    fft.add_function(wrap_pyfunction!(fft_1d_c2c, &fft)?)?;
    m.add_submodule(&fft)?;
    Ok(())
}
