# gpufft developer + release targets.

PWD := $(shell pwd)

.PHONY: build test wheel-dev wheel-manylinux clean-wheel

# Build the whole workspace (both backends).
build:
	cargo build --workspace

# CPU + GPU tests (requires a CUDA GPU and a Vulkan device).
test:
	cargo test --workspace

# Dev wheel: fast local build, NOT self-contained (the VkFFT shim .so is found
# via the cargo target dir, so it needs LD_LIBRARY_PATH). Use for local testing,
# not for distribution.
wheel-dev:
	cd gpufft-py && maturin build --release --auditwheel skip --out dist

# Release wheel: portable manylinux_2_28 wheel built in Docker. Bundles the
# VkFFT shim via auditwheel repair and excludes the runtime vendor libs. This
# is the wheel to attach to a GitHub Release (release.yml publishes it).
#
# `:z` relabels the bind mount for SELinux (Fedora/RHEL); without it the
# container is denied access to /work.
wheel-manylinux:
	docker build -f gpufft-py/Dockerfile.manylinux \
	    -t gpufft-builder:latest gpufft-py
	docker run --rm -v $(PWD):/work:z -w /work \
	    gpufft-builder:latest \
	    /work/gpufft-py/build-wheel.sh

clean-wheel:
	rm -rf gpufft-py/dist /tmp/wheel-raw
