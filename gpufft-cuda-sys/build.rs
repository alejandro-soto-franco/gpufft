//! Build script for gpufft-cuda-sys.
//!
//! Locates the CUDA Toolkit, runs bindgen against `cufft.h` and the narrow
//! slice of `cuda_runtime.h` that we use, and tells cargo to link against
//! `libcudart` and `libcufft`.
//!
//! CUDA discovery order:
//! 1. `CUDA_PATH` env var (if set and valid)
//! 2. `CUDA_HOME` env var (if set and valid)
//! 3. `/usr/local/cuda`
//! 4. `/opt/cuda`
//!
//! A readable `include/cufft.h` under the chosen prefix is the validation
//! criterion; anything else falls through to the next candidate.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let cuda_root = find_cuda_root().unwrap_or_else(|| {
        panic!(
            "CUDA Toolkit not found. Install the CUDA Toolkit or set \
             CUDA_PATH to point at a directory containing include/cufft.h."
        )
    });

    let include_dir = cuda_root.join("include");
    let lib_dir = cuda_root.join("lib64");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        // cuFFT
        .allowlist_type("cufftHandle")
        .allowlist_type("cufftResult(_t)?")
        .allowlist_type("cufftType(_t)?")
        .allowlist_type("cufftReal")
        .allowlist_type("cufftComplex")
        .allowlist_type("cufftDoubleReal")
        .allowlist_type("cufftDoubleComplex")
        .allowlist_function("cufftCreate")
        .allowlist_function("cufftDestroy")
        .allowlist_function("cufftPlan1d")
        .allowlist_function("cufftPlan2d")
        .allowlist_function("cufftPlan3d")
        .allowlist_function("cufftPlanMany")
        .allowlist_function("cufftExecC2C")
        .allowlist_function("cufftExecR2C")
        .allowlist_function("cufftExecC2R")
        .allowlist_function("cufftExecZ2Z")
        .allowlist_function("cufftExecD2Z")
        .allowlist_function("cufftExecZ2D")
        .allowlist_function("cufftSetStream")
        .allowlist_function("cufftGetVersion")
        .allowlist_var("CUFFT_.*")
        // CUDA Runtime (just enough for memory + streams)
        .allowlist_type("cudaError(_t|_enum)?")
        .allowlist_type("cudaMemcpyKind")
        .allowlist_type("cudaStream_t")
        .allowlist_type("CUstream_st")
        .allowlist_function("cudaMalloc")
        .allowlist_function("cudaFree")
        .allowlist_function("cudaMemcpy")
        .allowlist_function("cudaMemcpyAsync")
        .allowlist_function("cudaDeviceSynchronize")
        .allowlist_function("cudaStreamCreate")
        .allowlist_function("cudaStreamDestroy")
        .allowlist_function("cudaStreamSynchronize")
        .allowlist_function("cudaSetDevice")
        .allowlist_function("cudaGetDeviceCount")
        .allowlist_function("cudaGetErrorString")
        .allowlist_var("cudaSuccess")
        .layout_tests(false)
        .derive_default(true)
        .wrap_unsafe_ops(true)
        .generate()
        .expect("bindgen failed for cuFFT + CUDA Runtime");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}

fn find_cuda_root() -> Option<PathBuf> {
    let candidates = [
        env::var("CUDA_PATH").ok(),
        env::var("CUDA_HOME").ok(),
        Some("/usr/local/cuda".into()),
        Some("/opt/cuda".into()),
    ];

    for candidate in candidates.into_iter().flatten() {
        let path = PathBuf::from(&candidate);
        if is_cuda_root(&path) {
            return Some(path);
        }
    }

    None
}

fn is_cuda_root(p: &Path) -> bool {
    p.join("include/cufft.h").is_file() && p.join("lib64").is_dir()
}
