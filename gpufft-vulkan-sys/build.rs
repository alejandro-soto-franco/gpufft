//! Build script for gpufft-vulkan-sys.
//!
//! Compiles the VkFFT C shim against the vendored VkFFT headers and links
//! against the system Vulkan loader, glslang, and SPIRV-Tools.
//!
//! On Fedora, glslang ships statically and SPIRV-Tools dynamically.
//! Mixing static glslang with dynamic SPIRV-Tools inside the final Rust
//! binary caused C++ static-initializer crashes. The workaround is to
//! bundle the shim object and static glslang archives into a single
//! shared library, so all C++ dependencies resolve at .so link time.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=vkfft_shim.c");
    println!("cargo:rerun-if-changed=vendor/VkFFT/vkFFT/vkFFT.h");
    println!("cargo:rerun-if-changed=build.rs");

    let vkfft_include = PathBuf::from("vendor/VkFFT/vkFFT");
    if !vkfft_include.join("vkFFT.h").exists() {
        panic!(
            "vendored VkFFT not found at {}.\n\
             Run `git submodule update --init --recursive` or\n\
             clone DTolm/VkFFT into vendor/VkFFT.",
            vkfft_include.display()
        );
    }

    let vulkan = pkg_config::Config::new()
        .atleast_version("1.3")
        .probe("vulkan")
        .expect(
            "Vulkan SDK not found via pkg-config. Install `vulkan-headers` + \
             `vulkan-loader-devel` (Fedora) or `libvulkan-dev` (Debian/Ubuntu).",
        );

    let glslang_include = [
        PathBuf::from("/usr/include/glslang/Include"),
        PathBuf::from("/usr/local/include/glslang/Include"),
    ]
    .iter()
    .find(|p| p.join("glslang_c_interface.h").exists())
    .cloned()
    .expect(
        "glslang_c_interface.h not found. Install `glslang-devel` (Fedora) \
         or `glslang-dev` (Debian/Ubuntu).",
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Step 1: compile the shim to an object file.
    let obj_path = out_dir.join("vkfft_shim.o");
    let mut include_args: Vec<String> = vec![
        "-I.".into(),
        "-Ivendor/VkFFT".into(),
        format!("-I{}", vkfft_include.display()),
        format!("-I{}", glslang_include.display()),
        "-DVKFFT_BACKEND=0".into(),
    ];
    for p in &vulkan.include_paths {
        include_args.push(format!("-I{}", p.display()));
    }

    let status = Command::new("cc")
        .args(["-c", "-fPIC", "-O2", "-std=c11", "-w"])
        .args(&include_args)
        .arg("vkfft_shim.c")
        .arg("-o")
        .arg(&obj_path)
        .status()
        .expect("failed to run cc");
    assert!(status.success(), "cc failed to compile vkfft_shim.c");

    // Step 2: link shim + static glslang + dynamic SPIRV-Tools + vulkan into
    // a single shared library. See module docs for the ABI rationale.
    let lib_path = out_dir.join("libgpufft_vkfft_bundle.so");
    let status = Command::new("c++")
        .arg("-shared")
        .arg("-fPIC")
        .arg(&obj_path)
        .arg("-o")
        .arg(&lib_path)
        .args(["-Wl,--whole-archive", "-Wl,--allow-multiple-definition"])
        .args([
            "-lglslang",
            "-lMachineIndependent",
            "-lGenericCodeGen",
            "-lOSDependent",
            "-lSPIRV",
            "-lglslang-default-resource-limits",
        ])
        .args(["-Wl,--no-whole-archive"])
        .args(["-lSPIRV-Tools-shared", "-lSPIRV-Tools-opt", "-lvulkan"])
        .arg("-L/usr/lib64")
        .status()
        .expect("failed to run c++ for shared lib");
    assert!(
        status.success(),
        "c++ failed to create libgpufft_vkfft_bundle.so"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=gpufft_vkfft_bundle");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir.display());

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-Ivendor/VkFFT")
        .clang_arg("-Ivendor/VkFFT/vkFFT")
        .clang_arg(format!("-I{}", glslang_include.display()))
        .clang_arg("-DVKFFT_BACKEND=0")
        .clang_args(
            vulkan
                .include_paths
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_type("VkFFTApplication")
        .allowlist_type("VkFFTConfiguration")
        .allowlist_type("VkFFTLaunchParams")
        .allowlist_type("VkFFTResult.*")
        .allowlist_function("gpufft_vkfft_.*")
        .layout_tests(false)
        .derive_default(true)
        .wrap_unsafe_ops(true)
        .generate()
        .expect("bindgen failed");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
