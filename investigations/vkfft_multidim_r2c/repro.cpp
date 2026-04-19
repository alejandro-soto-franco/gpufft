// Minimal standalone repro:
//   Does VkFFT R2C with isInputFormatted=1 + isOutputFormatted=1 produce
//   the same spectrum as R2C with isInputFormatted=0 (padded in-place)?
//
// For 1D transforms: YES.
// For 3D transforms:  NO (observed L-inf diff of O(1) on a unit-scale input).
//
// Each mode runs the same R2C forward on the same physical real-space data.
// Mode A stages the input into a padded in-place buffer; mode B passes the
// tight real buffer directly via isInputFormatted + inputBufferStride. If
// both modes produce the same spectrum, the diff is ~1e-6 (float noise).
// If mode B is miscomputing the transform, the diff is ~O(1).
//
// Run on a local NVIDIA GPU (CUDA 13.0, driver 580, Vulkan 1.4), mode A
// agrees with FFTW on the spectrum up to float precision; mode B disagrees
// wildly for 3D cubic and non-cubic shapes alike.
//
// Build: `make`   Run: `./repro`

#define VKFFT_BACKEND 0

#include "vkFFT.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

// ===============================================================
//  Tiny VkResult + VkFFTResult check macros.
// ===============================================================

#define VK_CHECK(expr) do {                                           \
    VkResult _r = (expr);                                             \
    if (_r != VK_SUCCESS) {                                           \
        std::fprintf(stderr, "%s:%d: VkResult %d from %s\n",          \
            __FILE__, __LINE__, (int)_r, #expr);                      \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

#define VKFFT_CHECK(expr) do {                                        \
    VkFFTResult _r = (expr);                                          \
    if (_r != VKFFT_SUCCESS) {                                        \
        std::fprintf(stderr, "%s:%d: VkFFTResult %d from %s\n",       \
            __FILE__, __LINE__, (int)_r, #expr);                      \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

// ===============================================================
//  Minimal Vulkan context.
// ===============================================================

struct Ctx {
    VkInstance        instance = VK_NULL_HANDLE;
    VkPhysicalDevice  phys     = VK_NULL_HANDLE;
    VkDevice          device   = VK_NULL_HANDLE;
    VkQueue           queue    = VK_NULL_HANDLE;
    uint32_t          qfi      = 0;
    VkCommandPool     pool     = VK_NULL_HANDLE;
    VkFence           fence    = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
};

static void vulkan_init(Ctx& c) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &c.instance));

    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(c.instance, &n, nullptr));
    if (n == 0) { std::fprintf(stderr, "no Vulkan physical devices\n"); std::exit(1); }
    std::vector<VkPhysicalDevice> pds(n);
    VK_CHECK(vkEnumeratePhysicalDevices(c.instance, &n, pds.data()));
    c.phys = pds[0];
    for (auto pd : pds) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(pd, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { c.phys = pd; break; }
    }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, qs.data());
    for (uint32_t i = 0; i < qn; ++i) {
        if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qfi = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = c.qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(c.phys, &dci, nullptr, &c.device));
    vkGetDeviceQueue(c.device, c.qfi, 0, &c.queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = c.qfi;
    VK_CHECK(vkCreateCommandPool(c.device, &cpci, nullptr, &c.pool));

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(c.device, &fci, nullptr, &c.fence));

    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mem_props);
}

static void vulkan_destroy(Ctx& c) {
    vkDestroyFence(c.device, c.fence, nullptr);
    vkDestroyCommandPool(c.device, c.pool, nullptr);
    vkDestroyDevice(c.device, nullptr);
    vkDestroyInstance(c.instance, nullptr);
}

// ===============================================================
//  Buffer + transfer helpers.
// ===============================================================

static uint32_t find_memory_type(const Ctx& c, uint32_t filter, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.mem_props.memoryTypeCount; ++i) {
        if ((filter & (1u << i)) &&
            (c.mem_props.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    std::fprintf(stderr, "no matching memory type\n");
    std::exit(1);
}

struct Buf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; };

static Buf alloc_buffer(const Ctx& c, VkDeviceSize size,
                        VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Buf b{};
    VK_CHECK(vkCreateBuffer(c.device, &bci, nullptr, &b.buf));
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(c.device, b.buf, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory_type(c, mr.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(c.device, b.buf, b.mem, 0));
    return b;
}

static void free_buffer(const Ctx& c, Buf& b) {
    if (b.buf) vkDestroyBuffer(c.device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(c.device, b.mem, nullptr);
    b = {};
}

static void copy_buffer(const Ctx& c, VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBufferAllocateInfo a{};
    a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    a.commandPool = c.pool;
    a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    a.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &a, &cb));
    VkCommandBufferBeginInfo bb{};
    bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bb));
    VkBufferCopy region{}; region.size = size;
    vkCmdCopyBuffer(cb, src, dst, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkResetFences(c.device, 1, &c.fence);
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, c.fence));
    VK_CHECK(vkWaitForFences(c.device, 1, &c.fence, VK_TRUE, UINT64_MAX));
    vkFreeCommandBuffers(c.device, c.pool, 1, &cb);
}

static void upload(Ctx& c, VkBuffer dst, const void* src, VkDeviceSize size) {
    Buf stg = alloc_buffer(c, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* p = nullptr;
    VK_CHECK(vkMapMemory(c.device, stg.mem, 0, size, 0, &p));
    std::memcpy(p, src, size);
    vkUnmapMemory(c.device, stg.mem);
    copy_buffer(c, stg.buf, dst, size);
    free_buffer(c, stg);
}

static void download(Ctx& c, VkBuffer src, void* dst, VkDeviceSize size) {
    Buf stg = alloc_buffer(c, size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    copy_buffer(c, src, stg.buf, size);
    void* p = nullptr;
    VK_CHECK(vkMapMemory(c.device, stg.mem, 0, size, 0, &p));
    std::memcpy(dst, p, size);
    vkUnmapMemory(c.device, stg.mem);
    free_buffer(c, stg);
}

// ===============================================================
//  VkFFT execution helpers.
// ===============================================================

// Record `VkFFTAppend` into a one-shot command buffer and submit.
// Buffers to bind are specified via a launch-params filler callback.
template <typename Fill>
static void run_vkfft(Ctx& c, VkFFTApplication* app, int inverse, Fill fill) {
    VkCommandBufferAllocateInfo a{};
    a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    a.commandPool = c.pool;
    a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    a.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &a, &cb));
    VkCommandBufferBeginInfo bb{};
    bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bb));

    VkFFTLaunchParams lp{};
    lp.commandBuffer = &cb;
    fill(lp);
    VKFFT_CHECK(VkFFTAppend(app, inverse, &lp));

    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkResetFences(c.device, 1, &c.fence);
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, c.fence));
    VK_CHECK(vkWaitForFences(c.device, 1, &c.fence, VK_TRUE, UINT64_MAX));
    vkFreeCommandBuffers(c.device, c.pool, 1, &cb);
}

// ===============================================================
//  Mode A: padded in-place R2C (isInputFormatted = 0).
//
//  One buffer sized at 2*(Nx/2+1)*Ny*Nz floats. Host input is packed
//  into that layout with `padded` floats per innermost row. VkFFT reads
//  the padded real layout in place and writes the tight complex half
//  spectrum back into the same buffer.
// ===============================================================

static std::vector<float> mode_a_r2c(Ctx& c, uint32_t Nx, uint32_t Ny, uint32_t Nz,
                                     const std::vector<float>& tight_real) {
    const uint32_t half = Nx / 2 + 1;
    const uint32_t padded = 2 * half;
    const uint64_t buffer_bytes = (uint64_t)padded * Ny * Nz * sizeof(float);

    Buf inplace = alloc_buffer(c, buffer_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Pack tight input into padded layout on the host (one-off upload).
    std::vector<float> host_padded((size_t)padded * Ny * Nz, 0.0f);
    for (uint32_t z = 0; z < Nz; ++z) {
        for (uint32_t y = 0; y < Ny; ++y) {
            const size_t src = ((size_t)z * Ny + y) * Nx;
            const size_t dst = ((size_t)z * Ny + y) * padded;
            std::memcpy(&host_padded[dst], &tight_real[src], Nx * sizeof(float));
        }
    }
    upload(c, inplace.buf, host_padded.data(), buffer_bytes);

    // VkFFT config: in-place R2C, single buffer.
    uint64_t bufsize = buffer_bytes;
    VkFFTConfiguration cfg{};
    cfg.FFTdim = 3;
    cfg.size[0] = Nx;
    cfg.size[1] = Ny;
    cfg.size[2] = Nz;
    cfg.performR2C = 1;

    cfg.physicalDevice = &c.phys;
    cfg.device         = &c.device;
    cfg.queue          = &c.queue;
    cfg.commandPool    = &c.pool;
    cfg.fence          = &c.fence;

    cfg.buffer      = &inplace.buf;
    cfg.bufferSize  = &bufsize;
    cfg.bufferNum   = 1;

    VkFFTApplication app{};
    VKFFT_CHECK(initializeVkFFT(&app, cfg));

    run_vkfft(c, &app, /*inverse=*/0, [&](VkFFTLaunchParams& lp) {
        lp.buffer = &inplace.buf;
    });

    // The tight complex half-spectrum occupies the first Nx/2+1 complex
    // per row after R2C; copy out row-by-row to drop the padding tail.
    std::vector<float> tight_complex((size_t)half * Ny * Nz * 2, 0.0f);
    std::vector<float> host_after(host_padded.size());
    download(c, inplace.buf, host_after.data(), buffer_bytes);
    for (uint32_t z = 0; z < Nz; ++z) {
        for (uint32_t y = 0; y < Ny; ++y) {
            const size_t src = ((size_t)z * Ny + y) * padded;
            const size_t dst = ((size_t)z * Ny + y) * (size_t)half * 2;
            std::memcpy(&tight_complex[dst], &host_after[src],
                        (size_t)half * 2 * sizeof(float));
        }
    }

    deleteVkFFT(&app);
    free_buffer(c, inplace);
    return tight_complex;
}

// ===============================================================
//  Mode B: zero-copy out-of-place R2C via isInputFormatted = 1.
//
//  inputBuffer = tight real (Nx*Ny*Nz floats, no padding)
//  buffer      = tight complex half-spectrum ((Nx/2+1)*Ny*Nz complex)
//  Strides for inputBufferStride / bufferStride populated per VkFFT's
//  sample_15_precision_VkFFT_single_r2c.cpp reference (lines 347-350).
// ===============================================================

static std::vector<float> mode_b_r2c(Ctx& c, uint32_t Nx, uint32_t Ny, uint32_t Nz,
                                     const std::vector<float>& tight_real) {
    const uint32_t half = Nx / 2 + 1;
    const uint64_t real_bytes    = (uint64_t)Nx * Ny * Nz * sizeof(float);
    const uint64_t complex_bytes = (uint64_t)half * Ny * Nz * 2 * sizeof(float);

    Buf in = alloc_buffer(c, real_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf out = alloc_buffer(c, complex_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    upload(c, in.buf, tight_real.data(), real_bytes);

    uint64_t in_size  = real_bytes;
    uint64_t out_size = complex_bytes;

    VkFFTConfiguration cfg{};
    cfg.FFTdim = 3;
    cfg.size[0] = Nx;
    cfg.size[1] = Ny;
    cfg.size[2] = Nz;
    cfg.performR2C      = 1;
    cfg.isInputFormatted = 1;  // tight (non-padded) real input in a separate buffer

    // Non-padded row strides for the tight real input, matching sample_15.
    cfg.inputBufferStride[0] = Nx;
    cfg.inputBufferStride[1] = (uint64_t)Nx * Ny;
    cfg.inputBufferStride[2] = (uint64_t)Nx * Ny * Nz;
    // bufferStride defaults (size[0]/2+1, (Nx/2+1)*Ny, ...) are used for
    // the tight complex half-spectrum output.

    cfg.physicalDevice = &c.phys;
    cfg.device         = &c.device;
    cfg.queue          = &c.queue;
    cfg.commandPool    = &c.pool;
    cfg.fence          = &c.fence;

    cfg.inputBuffer      = &in.buf;
    cfg.inputBufferSize  = &in_size;
    cfg.inputBufferNum   = 1;
    cfg.buffer           = &out.buf;
    cfg.bufferSize       = &out_size;
    cfg.bufferNum        = 1;

    VkFFTApplication app{};
    VKFFT_CHECK(initializeVkFFT(&app, cfg));

    run_vkfft(c, &app, /*inverse=*/0, [&](VkFFTLaunchParams& lp) {
        lp.inputBuffer = &in.buf;
        lp.buffer      = &out.buf;
    });

    std::vector<float> tight_complex((size_t)half * Ny * Nz * 2, 0.0f);
    download(c, out.buf, tight_complex.data(), complex_bytes);

    deleteVkFFT(&app);
    free_buffer(c, in);
    free_buffer(c, out);
    return tight_complex;
}

// ===============================================================
//  Mode C: isInputFormatted = 1, but bind SCRATCH buffers at init and
//  override with the real user buffers via VkFFTLaunchParams at execute.
//
//  This is the pattern a language-binding plan cache typically wants:
//  the plan is built once at init time (before the user allocates their
//  working buffers), and the actual buffer handles are supplied per
//  execute. VkFFT exposes `inputBuffer` / `buffer` / `outputBuffer` on
//  VkFFTLaunchParams for exactly this use.
// ===============================================================

static std::vector<float> mode_c_r2c(Ctx& c, uint32_t Nx, uint32_t Ny, uint32_t Nz,
                                     const std::vector<float>& tight_real) {
    const uint32_t half = Nx / 2 + 1;
    const uint64_t real_bytes    = (uint64_t)Nx * Ny * Nz * sizeof(float);
    const uint64_t complex_bytes = (uint64_t)half * Ny * Nz * 2 * sizeof(float);

    // Scratch buffers bound at VkFFT init (never touched at runtime).
    Buf scratch_in = alloc_buffer(c, real_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf scratch_out = alloc_buffer(c, complex_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Real user buffers, allocated separately.
    Buf user_in = alloc_buffer(c, real_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf user_out = alloc_buffer(c, complex_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    upload(c, user_in.buf, tight_real.data(), real_bytes);

    uint64_t in_size  = real_bytes;
    uint64_t out_size = complex_bytes;

    VkFFTConfiguration cfg{};
    cfg.FFTdim = 3;
    cfg.size[0] = Nx;
    cfg.size[1] = Ny;
    cfg.size[2] = Nz;
    cfg.performR2C      = 1;
    cfg.isInputFormatted = 1;
    cfg.inputBufferStride[0] = Nx;
    cfg.inputBufferStride[1] = (uint64_t)Nx * Ny;
    cfg.inputBufferStride[2] = (uint64_t)Nx * Ny * Nz;

    cfg.physicalDevice = &c.phys;
    cfg.device         = &c.device;
    cfg.queue          = &c.queue;
    cfg.commandPool    = &c.pool;
    cfg.fence          = &c.fence;

    // Bind SCRATCH handles at init time.
    cfg.inputBuffer      = &scratch_in.buf;
    cfg.inputBufferSize  = &in_size;
    cfg.inputBufferNum   = 1;
    cfg.buffer           = &scratch_out.buf;
    cfg.bufferSize       = &out_size;
    cfg.bufferNum        = 1;

    VkFFTApplication app{};
    VKFFT_CHECK(initializeVkFFT(&app, cfg));

    // Override with USER handles at launch time.
    run_vkfft(c, &app, /*inverse=*/0, [&](VkFFTLaunchParams& lp) {
        lp.inputBuffer = &user_in.buf;
        lp.buffer      = &user_out.buf;
    });

    std::vector<float> tight_complex((size_t)half * Ny * Nz * 2, 0.0f);
    download(c, user_out.buf, tight_complex.data(), complex_bytes);

    deleteVkFFT(&app);
    free_buffer(c, user_out);
    free_buffer(c, user_in);
    free_buffer(c, scratch_out);
    free_buffer(c, scratch_in);
    return tight_complex;
}

// ===============================================================
//  Mode D: R2C + C2R round-trip via the plan-cache pattern.
//
//  Separate R2C and C2R plans, each configured with scratch buffers at
//  init and user buffers overridden via VkFFTLaunchParams at execute.
//  This mirrors what a Rust backend does when `plan_r2c` and `plan_c2r`
//  are distinct objects with no shared VkFFT app.
//
//  Returns the real output after `c2r(r2c(input))`. Expected: host * N,
//  where N = Nx * Ny * Nz (cuFFT / VkFFT unnormalised convention).
// ===============================================================

static std::vector<float> mode_d_roundtrip(Ctx& c, uint32_t Nx, uint32_t Ny, uint32_t Nz,
                                           const std::vector<float>& tight_real) {
    const uint32_t half = Nx / 2 + 1;
    const uint64_t real_bytes    = (uint64_t)Nx * Ny * Nz * sizeof(float);
    const uint64_t complex_bytes = (uint64_t)half * Ny * Nz * 2 * sizeof(float);

    // User buffers.
    Buf user_real    = alloc_buffer(c, real_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf user_complex = alloc_buffer(c, complex_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf user_real_back = alloc_buffer(c, real_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // R2C scratches.
    Buf r2c_si = alloc_buffer(c, real_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf r2c_so = alloc_buffer(c, complex_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // C2R scratches + intermediate.
    Buf c2r_si   = alloc_buffer(c, complex_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf c2r_mid  = alloc_buffer(c, complex_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf c2r_so   = alloc_buffer(c, real_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    upload(c, user_real.buf, tight_real.data(), real_bytes);

    uint64_t in_size_r  = real_bytes;
    uint64_t out_size_r = complex_bytes;
    uint64_t in_size_c  = complex_bytes;
    uint64_t mid_size_c = complex_bytes;
    uint64_t out_size_c = real_bytes;

    // --- R2C plan ---
    VkFFTConfiguration cfg_r{};
    cfg_r.FFTdim = 3;
    cfg_r.size[0] = Nx; cfg_r.size[1] = Ny; cfg_r.size[2] = Nz;
    cfg_r.performR2C = 1;
    cfg_r.isInputFormatted = 1;
    cfg_r.inputBufferStride[0] = Nx;
    cfg_r.inputBufferStride[1] = (uint64_t)Nx * Ny;
    cfg_r.inputBufferStride[2] = (uint64_t)Nx * Ny * Nz;
    cfg_r.physicalDevice = &c.phys;
    cfg_r.device         = &c.device;
    cfg_r.queue          = &c.queue;
    cfg_r.commandPool    = &c.pool;
    cfg_r.fence          = &c.fence;
    cfg_r.inputBuffer      = &r2c_si.buf;
    cfg_r.inputBufferSize  = &in_size_r;
    cfg_r.inputBufferNum   = 1;
    cfg_r.buffer           = &r2c_so.buf;
    cfg_r.bufferSize       = &out_size_r;
    cfg_r.bufferNum        = 1;

    VkFFTApplication app_r{};
    VKFFT_CHECK(initializeVkFFT(&app_r, cfg_r));

    // --- C2R plan ---
    VkFFTConfiguration cfg_c{};
    cfg_c.FFTdim = 3;
    cfg_c.size[0] = Nx; cfg_c.size[1] = Ny; cfg_c.size[2] = Nz;
    cfg_c.performR2C = 1;
    cfg_c.isInputFormatted = 1;
    cfg_c.isOutputFormatted = 1;
    cfg_c.outputBufferStride[0] = Nx;
    cfg_c.outputBufferStride[1] = (uint64_t)Nx * Ny;
    cfg_c.outputBufferStride[2] = (uint64_t)Nx * Ny * Nz;
    cfg_c.physicalDevice = &c.phys;
    cfg_c.device         = &c.device;
    cfg_c.queue          = &c.queue;
    cfg_c.commandPool    = &c.pool;
    cfg_c.fence          = &c.fence;
    cfg_c.inputBuffer      = &c2r_si.buf;
    cfg_c.inputBufferSize  = &in_size_c;
    cfg_c.inputBufferNum   = 1;
    cfg_c.buffer           = &c2r_mid.buf;
    cfg_c.bufferSize       = &mid_size_c;
    cfg_c.bufferNum        = 1;
    cfg_c.outputBuffer     = &c2r_so.buf;
    cfg_c.outputBufferSize = &out_size_c;
    cfg_c.outputBufferNum  = 1;

    VkFFTApplication app_c{};
    VKFFT_CHECK(initializeVkFFT(&app_c, cfg_c));

    // R2C: user_real -> user_complex.
    run_vkfft(c, &app_r, /*inverse=*/0, [&](VkFFTLaunchParams& lp) {
        lp.inputBuffer = &user_real.buf;
        lp.buffer      = &user_complex.buf;
    });

    // C2R: user_complex -> user_real_back (through c2r_mid).
    run_vkfft(c, &app_c, /*inverse=*/1, [&](VkFFTLaunchParams& lp) {
        lp.inputBuffer  = &user_complex.buf;
        lp.buffer       = &c2r_mid.buf;
        lp.outputBuffer = &user_real_back.buf;
    });

    std::vector<float> host_back((size_t)Nx * Ny * Nz, 0.0f);
    download(c, user_real_back.buf, host_back.data(), real_bytes);

    deleteVkFFT(&app_c);
    deleteVkFFT(&app_r);
    free_buffer(c, c2r_so);
    free_buffer(c, c2r_mid);
    free_buffer(c, c2r_si);
    free_buffer(c, r2c_so);
    free_buffer(c, r2c_si);
    free_buffer(c, user_real_back);
    free_buffer(c, user_complex);
    free_buffer(c, user_real);
    return host_back;
}

// ===============================================================
//  Compare.
// ===============================================================

struct Stats { float linf = 0.0f; float l2_rel = 0.0f; };

static Stats compare(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "size mismatch: %zu vs %zu\n", a.size(), b.size());
        std::exit(1);
    }
    double num = 0.0, den = 0.0;
    float linf = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        linf = std::fmax(linf, std::fabs(d));
        num += (double)d * d;
        den += (double)a[i] * a[i];
    }
    Stats s;
    s.linf = linf;
    s.l2_rel = den > 0.0 ? (float)std::sqrt(num / den) : (float)std::sqrt(num);
    return s;
}

static std::vector<float> sine_input(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = std::sin(0.17f * (float)i);
    return v;
}

// ===============================================================
//  Main.
// ===============================================================

static void run_case(Ctx& c, uint32_t Nx, uint32_t Ny, uint32_t Nz) {
    std::printf("-- Shape %ux%ux%u (Nx innermost) --\n", Nx, Ny, Nz);

    auto input = sine_input((size_t)Nx * Ny * Nz);
    auto out_a = mode_a_r2c(c, Nx, Ny, Nz, input);
    auto out_b = mode_b_r2c(c, Nx, Ny, Nz, input);
    auto out_c = mode_c_r2c(c, Nx, Ny, Nz, input);

    auto sab = compare(out_a, out_b);
    auto sac = compare(out_a, out_c);
    double max_a = 0.0;
    for (float x : out_a) max_a = std::fmax(max_a, std::fabs(x));

    std::printf("  padded in-place (A):            ||.||_inf = %.6f\n", max_a);
    std::printf("  formatted, user at init (B):    diff vs A  L-inf = %.6f, L2_rel = %.6e  %s\n",
                sab.linf, sab.l2_rel,
                sab.linf < 1e-3f ? "[OK]" : "[DIFFER]");
    std::printf("  formatted, launch-override (C): diff vs A  L-inf = %.6f, L2_rel = %.6e  %s\n",
                sac.linf, sac.l2_rel,
                sac.linf < 1e-3f ? "[OK]" : "[DIFFER]");

    // Round-trip via the plan-cache pattern (two separate plans with
    // launch-param overrides). Expected: c2r(r2c(input)) == input * N.
    auto roundtrip = mode_d_roundtrip(c, Nx, Ny, Nz, input);
    const double N = (double)Nx * Ny * Nz;
    double linf_rt = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        const double diff = (double)roundtrip[i] / N - (double)input[i];
        linf_rt = std::fmax(linf_rt, std::fabs(diff));
    }
    std::printf("  round-trip R2C+C2R (D):         L-inf(back/N - input) = %.6e  %s\n\n",
                linf_rt, linf_rt < 1e-3 ? "[OK]" : "[DIFFER]");
}

int main() {
    Ctx c;
    vulkan_init(c);

    std::printf("VkFFT version: %d\n", VkFFTGetVersion());
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(c.phys, &p);
    std::printf("Device: %s\n\n", p.deviceName);

    // 1D: expected to agree.
    run_case(c, 1024, 1, 1);
    // 2D cubic: probe.
    run_case(c, 64, 64, 1);
    // 3D cubic: expected to disagree.
    run_case(c, 32, 32, 32);
    // 3D non-cubic: expected to disagree.
    run_case(c, 16, 32, 64);

    vulkan_destroy(c);
    return 0;
}
