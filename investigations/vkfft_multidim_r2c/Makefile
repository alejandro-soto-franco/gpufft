# Build the VkFFT multi-dim R2C repro against the VkFFT submodule vendored
# inside gpufft-vulkan-sys. Assumes Vulkan headers, glslang, and SPIRV-Tools
# are installed system-wide (same prerequisites as gpufft itself).
#
#   Fedora:  sudo dnf install vulkan-headers vulkan-loader-devel \
#                             glslang-devel spirv-tools-devel
#   Debian/Ubuntu: install via LunarG Vulkan SDK.

VKFFT_ROOT ?= $(abspath ../../gpufft-vulkan-sys/vendor/VkFFT)

CXX      ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wno-missing-field-initializers \
            -DVKFFT_BACKEND=0 \
            -I$(VKFFT_ROOT) \
            -I$(VKFFT_ROOT)/vkFFT \
            -I/usr/include/glslang/Include

# Static glslang + dynamic SPIRV-Tools: match the link recipe used by
# gpufft-vulkan-sys/build.rs.
LDFLAGS ?= -Wl,--whole-archive -Wl,--allow-multiple-definition \
           -lglslang -lMachineIndependent -lGenericCodeGen \
           -lOSDependent -lSPIRV -lglslang-default-resource-limits \
           -Wl,--no-whole-archive \
           -lSPIRV-Tools-shared -lSPIRV-Tools-opt \
           -lvulkan

repro: repro.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f repro

.PHONY: clean
