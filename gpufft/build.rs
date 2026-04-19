//! Build script for the `gpufft` crate.
//!
//! Compiles the Vulkan compute shaders under `shaders/` to SPIR-V via
//! `glslangValidator` when the `vulkan` feature is enabled, and writes
//! the binaries into `OUT_DIR` so they can be `include_bytes!`'d by the
//! Vulkan backend.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Only compile shaders when the Vulkan backend is enabled; other
    // feature combinations do not need them.
    if env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=shaders/stride_copy.comp");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_path = out_dir.join("stride_copy.spv");

    let status = Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.2"])
        .arg("shaders/stride_copy.comp")
        .arg("-o")
        .arg(&out_path)
        .status()
        .expect(
            "failed to run glslangValidator. Install `glslang-devel` \
             (Fedora) or `glslang-dev` (Debian/Ubuntu), or install the \
             LunarG Vulkan SDK.",
        );
    assert!(
        status.success(),
        "glslangValidator failed to compile stride_copy.comp"
    );
}
