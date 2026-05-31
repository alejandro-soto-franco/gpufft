#!/usr/bin/env bash
# Build the gpufft manylinux_2_28_x86_64 wheel inside the image produced by
# Dockerfile.manylinux. Assumes /work is mounted to the repo root.
#
# The wheel links cuFFT (NVIDIA) and embeds the VkFFT shim. auditwheel repair
# bundles our own shim + its portable C++ deps (SPIRV-Tools, glslang is static)
# into gpufft.libs/, but EXCLUDES the runtime-provided vendor libraries so the
# wheel stays small and uses the user's driver/SDK:
#   - libvulkan.so.1  (Vulkan loader, from the user's ICD/driver)
#   - libcufft / libcudart / libcuda (from the user's CUDA install + driver)

set -euo pipefail

cd /work/gpufft-py

# /work/target is the host cargo-targets symlink, which dangles in the
# container; point cargo at a real in-container directory.
export CARGO_TARGET_DIR=/tmp/gpufft-target

# Build an unrepaired wheel first (skip), then repair with explicit excludes.
echo "Building wheel (maturin, auditwheel skip)..."
"$MATURIN" build --release \
    --compatibility manylinux_2_28 \
    --auditwheel skip \
    --out /tmp/wheel-raw

RAW=$(ls /tmp/wheel-raw/gpufft-*.whl | head -1)
echo "Raw wheel: $RAW"

echo "Repairing wheel (bundle VkFFT shim, exclude vendor libs)..."
auditwheel repair "$RAW" \
    --plat manylinux_2_28_x86_64 \
    --exclude libvulkan.so.1 \
    --exclude 'libcufft.so*' \
    --exclude 'libcudart.so*' \
    --exclude libcuda.so.1 \
    --exclude 'libnvrtc.so*' \
    -w /work/dist

echo
echo "Wheel(s) produced:"
ls -la /work/dist/
echo
echo "auditwheel show:"
auditwheel show /work/dist/gpufft-*manylinux_2_28*.whl || true
