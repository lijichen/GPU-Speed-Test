// gpu_stress_nccl_g2g_link.cu
// Build: nvcc -O3 -std=c++17 gpu_stress_nccl_g2g_link.cu -o gpu_stress_nccl -ldl -Xcompiler -pthread
// Runtime: ./gpu_stress_nccl [--single-seconds 30] [--all-seconds <sec>] [--reserve-mb 512] [--nccl-mb 256] [--nccl-batch 4] [--temp-threshold 80] [--g2g-seconds <sec>] [--g2g-mb 512] [--g2g-aggregate-seconds <sec>]
//
// This program does NOT require NCCL/NVML headers or linking with -lnccl/-lnvidia-ml.
// It tries to dlopen libnccl.so.2/libnccl.so and libnvidia-ml.so.1/libnvidia-ml.so
// at runtime. If NCCL is available, it runs an NCCL AllReduce benchmark after CUDA
// memory stress tests. It also benchmarks directed GPU-to-GPU pair bandwidth using
// CUDA P2P/cudaMemcpyPeer, peer read/write kernels, optional NCCL Send/Recv, and
// pinned host-staged fallback. It also labels GPU-to-GPU link paths using nvidia-smi topo -m when available. If NVML is available, it monitors GPU temperature while tests run.

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdio>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

static constexpr size_t KiB = 1024ull;
static constexpr size_t MiB = 1024ull * KiB;
static constexpr size_t GiB = 1024ull * MiB;

static std::mutex gPrintMutex;

static std::string bytesToGiB(size_t bytes) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(2) << (double)bytes / (double)GiB << " GiB";
  return os.str();
}

static std::string cudaErrorString(cudaError_t e) {
  std::ostringstream os;
  os << cudaGetErrorName(e) << ": " << cudaGetErrorString(e);
  return os.str();
}

struct Options {
  double singleSeconds = 30.0;
  double allSeconds = 0.0;  // 0 means: use singleSeconds.
  size_t reserveBytes = 512ull * MiB;
  size_t ncclBytes = 256ull * MiB;
  int ncclBatch = 4;
  bool tempMonitor = true;
  double tempThresholdC = 80.0;
  int tempIntervalMs = 1000;

  bool g2gTest = true;
  bool g2gPairTest = true;
  double g2gSeconds = 0.0;  // 0 means: min(1 sec, effectiveAllSeconds()).
  size_t g2gBytes = 512ull * MiB;
  int g2gBatch = 4;
  bool g2gHostAlways = false;
  bool g2gBidirectionalPair = false;

  // Aggregate tests try to measure the per-GPU NVLink/NVSwitch budget by
  // launching many GPU-to-GPU copies concurrently. 0 means: min(1 sec,
  // effectiveG2GSeconds()) so pair tests can run long without making the
  // aggregate section unexpectedly expensive.
  bool g2gAggregateTest = true;
  double g2gAggregateSeconds = 0.0;

  double effectiveAllSeconds() const {
    return allSeconds > 0.0 ? allSeconds : singleSeconds;
  }

  double effectiveG2GSeconds() const {
    const double all = effectiveAllSeconds();
    return g2gSeconds > 0.0 ? g2gSeconds : std::min(1.0, all);
  }

  double effectiveG2GAggregateSeconds() const {
    const double pairSeconds = effectiveG2GSeconds();
    return g2gAggregateSeconds > 0.0 ? g2gAggregateSeconds : std::min(1.0, pairSeconds);
  }
};

static void printUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "Options:\n"
      << "  --single-seconds <sec> Seconds for each single-GPU memory stress test. Default: 30\n"
      << "  --single-gpu-seconds <sec> Alias for --single-seconds\n"
      << "  --all-seconds <sec>    Seconds for concurrent all-GPU memory stress and NCCL link test. Default: same as --single-seconds\n"
      << "  --all-gpu-seconds <sec> Alias for --all-seconds\n"
      << "  --concurrent-seconds <sec> Alias for --all-seconds\n"
      << "  --seconds <sec>        Backward-compatible alias for --single-seconds\n"
      << "  --reserve-mb <MB>     Keep this much free memory per GPU for CUDA/NCCL/OS. Default: 512\n"
      << "  --nccl-mb <MB>        NCCL send buffer size per GPU. It uses the same size for recv. Default: 256\n"
      << "  --nccl-batch <N>      Queue N AllReduce ops before synchronizing. Default: 4\n"
      << "  --temp-threshold <C> Warn when any GPU temperature reaches/exceeds this value. Default: 80\n"
      << "  --temp-interval-ms <N> Poll GPU temperature every N milliseconds. Default: 1000\n"
      << "  --no-temp-monitor    Disable GPU temperature monitoring\n"
      << "  --g2g-seconds <sec>  Seconds for each method in each directed GPU-to-GPU pair. Default: min(1, all-GPU seconds)\n"
      << "  --peer-seconds <sec> Alias for --g2g-seconds\n"
      << "  --g2g-mb <MB>        Buffer size for each GPU-to-GPU pair test. Default: 512\n"
      << "  --peer-mb <MB>       Alias for --g2g-mb\n"
      << "  --g2g-batch <N>      Queue N pair transfers/kernels before synchronizing. Default: 4\n"
      << "  --g2g-host-always    Also benchmark pinned host-staged path even when CUDA P2P is available\n"
      << "  --g2g-bidir-pair     Also measure simultaneous bidirectional cudaMemcpyPeer for each GPU pair\n"
      << "  --no-g2g-pair-test   Disable directed pair matrix tests, but keep aggregate tests if enabled\n"
      << "  --g2g-aggregate-seconds <sec> Seconds for per-GPU aggregate one-to-all/all-to-one/bidir tests. Default: min(1, g2g seconds)\n"
      << "  --no-g2g-aggregate-test Disable per-GPU aggregate GPU-to-GPU tests\n"
      << "  --no-g2g-test        Disable pairwise and aggregate GPU-to-GPU bandwidth tests\n"
      << "  --help                Show this help\n";
}

static bool parseArgs(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto needValue = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else if (a == "--single-seconds" || a == "--single-gpu-seconds" || a == "--seconds") {
      const char* v = needValue(a.c_str());
      if (!v) return false;
      opt.singleSeconds = std::stod(v);
      if (opt.singleSeconds <= 0.0) return false;
    } else if (a == "--all-seconds" || a == "--all-gpu-seconds" || a == "--concurrent-seconds") {
      const char* v = needValue(a.c_str());
      if (!v) return false;
      opt.allSeconds = std::stod(v);
      if (opt.allSeconds <= 0.0) return false;
    } else if (a == "--reserve-mb") {
      const char* v = needValue("--reserve-mb");
      if (!v) return false;
      opt.reserveBytes = (size_t)std::stoull(v) * MiB;
    } else if (a == "--nccl-mb") {
      const char* v = needValue("--nccl-mb");
      if (!v) return false;
      opt.ncclBytes = (size_t)std::stoull(v) * MiB;
    } else if (a == "--nccl-batch") {
      const char* v = needValue("--nccl-batch");
      if (!v) return false;
      opt.ncclBatch = std::stoi(v);
      if (opt.ncclBatch <= 0) return false;
    } else if (a == "--temp-threshold") {
      const char* v = needValue("--temp-threshold");
      if (!v) return false;
      opt.tempThresholdC = std::stod(v);
      if (opt.tempThresholdC <= 0.0) return false;
    } else if (a == "--temp-interval-ms") {
      const char* v = needValue("--temp-interval-ms");
      if (!v) return false;
      opt.tempIntervalMs = std::stoi(v);
      if (opt.tempIntervalMs <= 0) return false;
    } else if (a == "--no-temp-monitor") {
      opt.tempMonitor = false;
    } else if (a == "--g2g-seconds" || a == "--peer-seconds" || a == "--p2p-seconds") {
      const char* v = needValue(a.c_str());
      if (!v) return false;
      opt.g2gSeconds = std::stod(v);
      if (opt.g2gSeconds <= 0.0) return false;
    } else if (a == "--g2g-mb" || a == "--peer-mb" || a == "--p2p-mb") {
      const char* v = needValue(a.c_str());
      if (!v) return false;
      opt.g2gBytes = (size_t)std::stoull(v) * MiB;
      if (opt.g2gBytes < MiB) return false;
    } else if (a == "--g2g-batch" || a == "--peer-batch" || a == "--p2p-batch") {
      const char* v = needValue(a.c_str());
      if (!v) return false;
      opt.g2gBatch = std::stoi(v);
      if (opt.g2gBatch <= 0) return false;
    } else if (a == "--g2g-host-always" || a == "--peer-host-always" || a == "--p2p-host-always") {
      opt.g2gHostAlways = true;
    } else if (a == "--g2g-bidir-pair" || a == "--peer-bidir-pair" || a == "--p2p-bidir-pair") {
      opt.g2gBidirectionalPair = true;
    } else if (a == "--no-g2g-pair-test" || a == "--no-peer-pair-test" || a == "--no-p2p-pair-test") {
      opt.g2gPairTest = false;
    } else if (a == "--g2g-aggregate-seconds" || a == "--peer-aggregate-seconds" || a == "--p2p-aggregate-seconds") {
      const char* v = needValue(a.c_str());
      if (!v) return false;
      opt.g2gAggregateSeconds = std::stod(v);
      if (opt.g2gAggregateSeconds <= 0.0) return false;
    } else if (a == "--no-g2g-aggregate-test" || a == "--no-peer-aggregate-test" || a == "--no-p2p-aggregate-test") {
      opt.g2gAggregateTest = false;
    } else if (a == "--no-g2g-test" || a == "--no-peer-test" || a == "--no-p2p-test") {
      opt.g2gTest = false;
    } else {
      std::cerr << "Unknown option: " << a << "\n";
      return false;
    }
  }
  return true;
}

__global__ void memoryStressKernel(uint4* p, size_t n, unsigned int seed) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += stride) {
    uint4 v = p[i];
    v.x = v.x * 1664525u + 1013904223u + seed + (unsigned int)i;
    v.y ^= v.x + 0x9e3779b9u;
    v.z += (v.y ^ seed) + 0x85ebca6bu;
    v.w = (v.w << 1) | (v.w >> 31);
    p[i] = v;
  }
}

__global__ void peerRemoteReadKernel(const uint4* __restrict__ remoteSrc,
                                     uint4* __restrict__ localDst,
                                     size_t n, unsigned int seed) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += stride) {
    uint4 v = remoteSrc[i];  // remote GPU memory read
    v.x ^= seed + (unsigned int)i;
    v.y += 0x9e3779b9u;
    localDst[i] = v;         // local GPU memory write
  }
}

__global__ void peerRemoteWriteKernel(const uint4* __restrict__ localSrc,
                                      uint4* __restrict__ remoteDst,
                                      size_t n, unsigned int seed) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += stride) {
    uint4 v = localSrc[i];   // local GPU memory read
    v.z ^= seed + (unsigned int)i;
    v.w += 0x85ebca6bu;
    remoteDst[i] = v;        // remote GPU memory write
  }
}

static cudaError_t tryMallocShrinking(void** ptr, size_t& bytes, size_t minBytes, size_t alignBytes) {
  *ptr = nullptr;
  bytes = (bytes / alignBytes) * alignBytes;
  while (bytes >= minBytes) {
    cudaError_t e = cudaMalloc(ptr, bytes);
    if (e == cudaSuccess) return cudaSuccess;
    cudaGetLastError();  // clear sticky cudaMalloc error state
    bytes = (bytes * 95) / 100;
    bytes = (bytes / alignBytes) * alignBytes;
  }
  return cudaErrorMemoryAllocation;
}

struct MemResult {
  int device = -1;
  std::string name;
  size_t totalBytes = 0;
  size_t allocatedBytes = 0;
  int iterations = 0;
  double elapsedSec = 0.0;
  double gbps = 0.0;
  std::string error;
};

static MemResult runMemoryStressOneGpu(int dev, double seconds, size_t reserveBytes,
                                       std::atomic<int>* ready = nullptr,
                                       std::atomic<bool>* go = nullptr) {
  MemResult r;
  r.device = dev;

  cudaError_t e = cudaSetDevice(dev);
  if (e != cudaSuccess) {
    r.error = "cudaSetDevice failed: " + cudaErrorString(e);
    if (ready) ready->fetch_add(1);
    return r;
  }

  cudaDeviceProp prop{};
  e = cudaGetDeviceProperties(&prop, dev);
  if (e != cudaSuccess) {
    r.error = "cudaGetDeviceProperties failed: " + cudaErrorString(e);
    if (ready) ready->fetch_add(1);
    return r;
  }
  r.name = prop.name;
  r.totalBytes = prop.totalGlobalMem;

  // Create context before querying free memory.
  e = cudaFree(nullptr);
  if (e != cudaSuccess) {
    r.error = "cudaFree(nullptr) context init failed: " + cudaErrorString(e);
    if (ready) ready->fetch_add(1);
    return r;
  }

  size_t freeBytes = 0, totalBytes = 0;
  e = cudaMemGetInfo(&freeBytes, &totalBytes);
  if (e != cudaSuccess) {
    r.error = "cudaMemGetInfo failed: " + cudaErrorString(e);
    if (ready) ready->fetch_add(1);
    return r;
  }

  size_t targetBytes = 0;
  if (freeBytes > reserveBytes + 16 * MiB) {
    targetBytes = freeBytes - reserveBytes;
  } else {
    targetBytes = (freeBytes * 90) / 100;
  }
  targetBytes = (targetBytes / sizeof(uint4)) * sizeof(uint4);

  void* raw = nullptr;
  e = tryMallocShrinking(&raw, targetBytes, 16 * MiB, sizeof(uint4));
  if (e != cudaSuccess || raw == nullptr) {
    r.error = "cudaMalloc failed after shrinking target allocation: " + cudaErrorString(e);
    if (ready) ready->fetch_add(1);
    return r;
  }
  r.allocatedBytes = targetBytes;
  uint4* buf = reinterpret_cast<uint4*>(raw);
  const size_t elems = r.allocatedBytes / sizeof(uint4);

  cudaStream_t stream{};
  e = cudaStreamCreate(&stream);
  if (e != cudaSuccess) {
    r.error = "cudaStreamCreate failed: " + cudaErrorString(e);
    cudaFree(buf);
    if (ready) ready->fetch_add(1);
    return r;
  }

  e = cudaMemsetAsync(buf, 0xA5, r.allocatedBytes, stream);
  if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    r.error = "cudaMemsetAsync/sync failed: " + cudaErrorString(e);
    cudaStreamDestroy(stream);
    cudaFree(buf);
    if (ready) ready->fetch_add(1);
    return r;
  }

  const int threads = 256;
  const int blocks = std::max(1, std::min(prop.multiProcessorCount * 32, 65535));

  // Warmup.
  memoryStressKernel<<<blocks, threads, 0, stream>>>(buf, elems, 1234u);
  e = cudaPeekAtLastError();
  if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    r.error = "warmup kernel failed: " + cudaErrorString(e);
    cudaStreamDestroy(stream);
    cudaFree(buf);
    if (ready) ready->fetch_add(1);
    return r;
  }

  if (ready && go) {
    ready->fetch_add(1);
    while (!go->load(std::memory_order_acquire)) std::this_thread::yield();
  }

  const auto start = Clock::now();
  int iters = 0;
  double elapsed = 0.0;
  do {
    memoryStressKernel<<<blocks, threads, 0, stream>>>(buf, elems, 0xC001D00Du + (unsigned int)iters);
    e = cudaPeekAtLastError();
    if (e != cudaSuccess) {
      r.error = "kernel launch failed: " + cudaErrorString(e);
      break;
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
      r.error = "kernel execution failed: " + cudaErrorString(e);
      break;
    }
    ++iters;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < seconds);

  r.iterations = iters;
  r.elapsedSec = elapsed;
  if (r.error.empty() && elapsed > 0.0 && iters > 0) {
    // Kernel reads 16 bytes and writes 16 bytes per uint4 element.
    const double bytesMoved = (double)r.allocatedBytes * 2.0 * (double)iters;
    r.gbps = bytesMoved / elapsed / 1e9;
  }

  cudaStreamDestroy(stream);
  cudaFree(buf);
  return r;
}

using nvmlDevice_t_dyn = void*;
static constexpr int NVML_SUCCESS_DYN = 0;
static constexpr int NVML_TEMPERATURE_GPU_DYN = 0;

struct NvmlApi {
  void* handle = nullptr;
  bool initialized = false;
  std::string loadError;

  using InitFn = int (*)();
  using ShutdownFn = int (*)();
  using ErrorStringFn = const char* (*)(int);
  using GetHandleByPciBusIdFn = int (*)(const char*, nvmlDevice_t_dyn*);
  using GetTemperatureFn = int (*)(nvmlDevice_t_dyn, int, unsigned int*);

  InitFn nvmlInit = nullptr;
  ShutdownFn nvmlShutdown = nullptr;
  ErrorStringFn nvmlErrorString = nullptr;
  GetHandleByPciBusIdFn nvmlDeviceGetHandleByPciBusId = nullptr;
  GetTemperatureFn nvmlDeviceGetTemperature = nullptr;

  template <typename T>
  T getSym(const char* name) {
    return reinterpret_cast<T>(dlsym(handle, name));
  }

  bool load() {
    const char* libs[] = {"libnvidia-ml.so.1", "libnvidia-ml.so"};
    for (const char* lib : libs) {
      handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
      if (handle) break;
    }
    if (!handle) {
      const char* err = dlerror();
      loadError = err ? err : "dlopen(libnvidia-ml.so.1/libnvidia-ml.so) failed";
      return false;
    }

    nvmlInit = getSym<InitFn>("nvmlInit_v2");
    if (!nvmlInit) nvmlInit = getSym<InitFn>("nvmlInit");
    nvmlShutdown = getSym<ShutdownFn>("nvmlShutdown");
    nvmlErrorString = getSym<ErrorStringFn>("nvmlErrorString");
    nvmlDeviceGetHandleByPciBusId = getSym<GetHandleByPciBusIdFn>("nvmlDeviceGetHandleByPciBusId_v2");
    if (!nvmlDeviceGetHandleByPciBusId) {
      nvmlDeviceGetHandleByPciBusId = getSym<GetHandleByPciBusIdFn>("nvmlDeviceGetHandleByPciBusId");
    }
    nvmlDeviceGetTemperature = getSym<GetTemperatureFn>("nvmlDeviceGetTemperature");

    if (!nvmlInit || !nvmlShutdown || !nvmlDeviceGetHandleByPciBusId || !nvmlDeviceGetTemperature) {
      loadError = "missing required NVML symbol";
      return false;
    }

    int r = nvmlInit();
    if (r != NVML_SUCCESS_DYN) {
      loadError = "nvmlInit failed: " + err(r);
      return false;
    }
    initialized = true;
    return true;
  }

  std::string err(int code) const {
    if (nvmlErrorString) return std::string(nvmlErrorString(code));
    return std::string("NVML error code ") + std::to_string(code);
  }

  ~NvmlApi() {
    if (initialized && nvmlShutdown) nvmlShutdown();
    if (handle) dlclose(handle);
  }
};

struct TempStatus {
  bool available = false;
  bool hasReading = false;
  bool everOverThreshold = false;
  unsigned int lastC = 0;
  unsigned int maxC = 0;
  std::string error;
};

static bool initNvmlTemperatureHandles(NvmlApi& nvml, int deviceCount,
                                       std::vector<nvmlDevice_t_dyn>& handles,
                                       std::vector<TempStatus>& status,
                                       std::string& error) {
  handles.assign(deviceCount, nullptr);
  status.assign(deviceCount, TempStatus{});
  bool anyOk = false;

  for (int dev = 0; dev < deviceCount; ++dev) {
    char pciBusId[64] = {0};
    cudaError_t ce = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), dev);
    if (ce != cudaSuccess) {
      status[dev].error = "cudaDeviceGetPCIBusId failed: " + cudaErrorString(ce);
      continue;
    }

    nvmlDevice_t_dyn h = nullptr;
    int r = nvml.nvmlDeviceGetHandleByPciBusId(pciBusId, &h);
    if (r != NVML_SUCCESS_DYN || !h) {
      status[dev].error = std::string("nvmlDeviceGetHandleByPciBusId failed for PCI ") +
                          pciBusId + ": " + nvml.err(r);
      continue;
    }

    handles[dev] = h;
    status[dev].available = true;
    anyOk = true;
  }

  if (!anyOk) {
    error = "NVML loaded, but no CUDA GPU could be mapped to an NVML handle.";
    return false;
  }
  return true;
}

static void temperatureMonitorLoop(NvmlApi* nvml,
                                   const std::vector<nvmlDevice_t_dyn>* handles,
                                   const std::vector<cudaDeviceProp>* props,
                                   double thresholdC,
                                   int intervalMs,
                                   std::atomic<bool>* stop,
                                   std::vector<TempStatus>* status) {
  const int n = (int)handles->size();
  std::vector<bool> isOver(n, false);
  std::vector<Clock::time_point> lastWarn(n, Clock::time_point::min());
  const auto warnEvery = std::chrono::seconds(5);

  while (!stop->load(std::memory_order_acquire)) {
    const auto now = Clock::now();
    for (int dev = 0; dev < n; ++dev) {
      if ((*handles)[dev] == nullptr) continue;

      unsigned int tempC = 0;
      int r = nvml->nvmlDeviceGetTemperature((*handles)[dev], NVML_TEMPERATURE_GPU_DYN, &tempC);
      if (r == NVML_SUCCESS_DYN) {
        TempStatus& st = (*status)[dev];
        st.available = true;
        st.hasReading = true;
        st.lastC = tempC;
        st.maxC = std::max(st.maxC, tempC);

        if ((double)tempC >= thresholdC) {
          st.everOverThreshold = true;
          const bool shouldWarn = !isOver[dev] || lastWarn[dev] == Clock::time_point::min() ||
                                  (now - lastWarn[dev]) >= warnEvery;
          if (shouldWarn) {
            std::lock_guard<std::mutex> lock(gPrintMutex);
            std::cerr << "\n[TEMP WARNING] GPU " << dev << " (" << (*props)[dev].name << ") "
                      << tempC << " C >= threshold " << std::fixed << std::setprecision(1)
                      << thresholdC << " C\n";
            lastWarn[dev] = now;
          }
          isOver[dev] = true;
        } else {
          isOver[dev] = false;
        }
      } else {
        TempStatus& st = (*status)[dev];
        if (st.error.empty()) {
          st.error = "nvmlDeviceGetTemperature failed: " + nvml->err(r);
          std::lock_guard<std::mutex> lock(gPrintMutex);
          std::cerr << "\n[TEMP WARNING] GPU " << dev << " temperature query failed: "
                    << st.error << "\n";
        }
      }
    }

    int slept = 0;
    while (slept < intervalMs && !stop->load(std::memory_order_acquire)) {
      const int step = std::min(100, intervalMs - slept);
      std::this_thread::sleep_for(std::chrono::milliseconds(step));
      slept += step;
    }
  }
}

using ncclComm_t_dyn = void*;

// Values from NCCL public ABI: ncclFloat32 = 7, ncclSum = 0.
static constexpr int NCCL_DTYPE_FLOAT32 = 7;
static constexpr int NCCL_OP_SUM = 0;

struct NcclApi {
  void* handle = nullptr;
  int version = 0;

  using GetVersionFn = int (*)(int*);
  using GetErrorStringFn = const char* (*)(int);
  using CommInitAllFn = int (*)(ncclComm_t_dyn*, int, const int*);
  using CommDestroyFn = int (*)(ncclComm_t_dyn);
  using AllReduceFn = int (*)(const void*, void*, size_t, int, int, ncclComm_t_dyn, cudaStream_t);
  using SendFn = int (*)(const void*, size_t, int, int, ncclComm_t_dyn, cudaStream_t);
  using RecvFn = int (*)(void*, size_t, int, int, ncclComm_t_dyn, cudaStream_t);
  using GroupStartFn = int (*)();
  using GroupEndFn = int (*)();

  GetVersionFn ncclGetVersion = nullptr;
  GetErrorStringFn ncclGetErrorString = nullptr;
  CommInitAllFn ncclCommInitAll = nullptr;
  CommDestroyFn ncclCommDestroy = nullptr;
  AllReduceFn ncclAllReduce = nullptr;
  SendFn ncclSend = nullptr;
  RecvFn ncclRecv = nullptr;
  GroupStartFn ncclGroupStart = nullptr;
  GroupEndFn ncclGroupEnd = nullptr;

  std::string loadError;

  template <typename T>
  bool sym(T& fn, const char* name) {
    fn = reinterpret_cast<T>(dlsym(handle, name));
    if (!fn) {
      loadError = std::string("missing NCCL symbol: ") + name;
      return false;
    }
    return true;
  }

  bool load() {
    const char* libs[] = {"libnccl.so.2", "libnccl.so"};
    for (const char* lib : libs) {
      handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
      if (handle) break;
    }
    if (!handle) {
      const char* err = dlerror();
      loadError = err ? err : "dlopen(libnccl.so.2/libnccl.so) failed";
      return false;
    }

    if (!sym(ncclGetVersion, "ncclGetVersion")) return false;
    if (!sym(ncclGetErrorString, "ncclGetErrorString")) return false;
    if (!sym(ncclCommInitAll, "ncclCommInitAll")) return false;
    if (!sym(ncclCommDestroy, "ncclCommDestroy")) return false;
    if (!sym(ncclAllReduce, "ncclAllReduce")) return false;
    ncclSend = reinterpret_cast<SendFn>(dlsym(handle, "ncclSend"));
    ncclRecv = reinterpret_cast<RecvFn>(dlsym(handle, "ncclRecv"));
    if (!sym(ncclGroupStart, "ncclGroupStart")) return false;
    if (!sym(ncclGroupEnd, "ncclGroupEnd")) return false;

    int r = ncclGetVersion(&version);
    if (r != 0) {
      loadError = "ncclGetVersion failed";
      return false;
    }
    return true;
  }

  bool hasSendRecv() const {
    return ncclSend != nullptr && ncclRecv != nullptr;
  }

  std::string err(int code) const {
    if (ncclGetErrorString) return std::string(ncclGetErrorString(code));
    return std::string("NCCL error code ") + std::to_string(code);
  }

  ~NcclApi() {
    if (handle) dlclose(handle);
  }
};

struct NcclResult {
  bool available = false;
  bool ran = false;
  int version = 0;
  int nranks = 0;
  size_t bytesPerRank = 0;
  int iterations = 0;
  double elapsedSec = 0.0;
  double algGBps = 0.0;
  double busGBps = 0.0;
  std::string error;
};

static void destroyNcclComms(NcclApi& nccl, std::vector<ncclComm_t_dyn>& comms) {
  for (auto& c : comms) {
    if (c) {
      nccl.ncclCommDestroy(c);
      c = nullptr;
    }
  }
}

static void freeDeviceBuffers(const std::vector<int>& devs,
                              std::vector<void*>& send,
                              std::vector<void*>& recv) {
  for (size_t i = 0; i < devs.size(); ++i) {
    cudaSetDevice(devs[i]);
    if (send[i]) cudaFree(send[i]);
    if (recv[i]) cudaFree(recv[i]);
    send[i] = nullptr;
    recv[i] = nullptr;
  }
}

static bool allocateNcclBuffers(const std::vector<int>& devs, size_t& bytesPerRank,
                                std::vector<void*>& send, std::vector<void*>& recv,
                                std::string& error) {
  bytesPerRank = (bytesPerRank / sizeof(float)) * sizeof(float);
  while (bytesPerRank >= MiB) {
    bool ok = true;
    error.clear();
    for (size_t i = 0; i < devs.size(); ++i) {
      cudaError_t e = cudaSetDevice(devs[i]);
      if (e != cudaSuccess) {
        ok = false;
        error = "cudaSetDevice failed on GPU " + std::to_string(devs[i]) + ": " + cudaErrorString(e);
        break;
      }
      e = cudaMalloc(&send[i], bytesPerRank);
      if (e != cudaSuccess) {
        cudaGetLastError();
        ok = false;
        error = "cudaMalloc send buffer failed on GPU " + std::to_string(devs[i]) + ": " + cudaErrorString(e);
        break;
      }
      e = cudaMalloc(&recv[i], bytesPerRank);
      if (e != cudaSuccess) {
        cudaGetLastError();
        ok = false;
        error = "cudaMalloc recv buffer failed on GPU " + std::to_string(devs[i]) + ": " + cudaErrorString(e);
        break;
      }
      e = cudaMemset(send[i], 1, bytesPerRank);
      if (e == cudaSuccess) e = cudaMemset(recv[i], 0, bytesPerRank);
      if (e == cudaSuccess) e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        ok = false;
        error = "cudaMemset/sync NCCL buffer failed on GPU " + std::to_string(devs[i]) + ": " + cudaErrorString(e);
        break;
      }
    }
    if (ok) return true;
    freeDeviceBuffers(devs, send, recv);
    bytesPerRank /= 2;
    bytesPerRank = (bytesPerRank / sizeof(float)) * sizeof(float);
  }
  if (error.empty()) error = "NCCL buffer allocation failed; buffer dropped below 1 MiB";
  return false;
}

static bool syncAllStreams(const std::vector<int>& devs, const std::vector<cudaStream_t>& streams,
                           std::string& error) {
  for (size_t i = 0; i < devs.size(); ++i) {
    cudaSetDevice(devs[i]);
    cudaError_t e = cudaStreamSynchronize(streams[i]);
    if (e != cudaSuccess) {
      error = "cudaStreamSynchronize failed on GPU " + std::to_string(devs[i]) + ": " + cudaErrorString(e);
      return false;
    }
  }
  return true;
}

static bool launchAllReduceBatch(NcclApi& nccl, const std::vector<int>& devs,
                                 const std::vector<ncclComm_t_dyn>& comms,
                                 const std::vector<void*>& send,
                                 const std::vector<void*>& recv,
                                 const std::vector<cudaStream_t>& streams,
                                 size_t countFloats, int batch, std::string& error) {
  for (int b = 0; b < batch; ++b) {
    int r = nccl.ncclGroupStart();
    if (r != 0) {
      error = "ncclGroupStart failed: " + nccl.err(r);
      return false;
    }
    for (size_t i = 0; i < devs.size(); ++i) {
      cudaSetDevice(devs[i]);
      r = nccl.ncclAllReduce(send[i], recv[i], countFloats, NCCL_DTYPE_FLOAT32, NCCL_OP_SUM,
                             comms[i], streams[i]);
      if (r != 0) {
        error = "ncclAllReduce enqueue failed on GPU " + std::to_string(devs[i]) + ": " + nccl.err(r);
        // Try to close the group before returning.
        nccl.ncclGroupEnd();
        return false;
      }
    }
    r = nccl.ncclGroupEnd();
    if (r != 0) {
      error = "ncclGroupEnd failed: " + nccl.err(r);
      return false;
    }
  }

  return syncAllStreams(devs, streams, error);
}

static NcclResult runNcclAllReduce(NcclApi& nccl, int deviceCount, const Options& opt) {
  NcclResult res;
  res.available = true;
  res.version = nccl.version;
  res.nranks = deviceCount;

  if (deviceCount < 2) {
    res.error = "Only one CUDA GPU is visible; NCCL link test needs at least 2 GPUs.";
    return res;
  }

  std::vector<int> devs(deviceCount);
  std::iota(devs.begin(), devs.end(), 0);

  std::vector<cudaStream_t> streams(deviceCount, nullptr);
  std::vector<void*> send(deviceCount, nullptr), recv(deviceCount, nullptr);
  std::vector<ncclComm_t_dyn> comms(deviceCount, nullptr);

  for (int i = 0; i < deviceCount; ++i) {
    cudaError_t e = cudaSetDevice(devs[i]);
    if (e != cudaSuccess) {
      res.error = "cudaSetDevice failed on GPU " + std::to_string(i) + ": " + cudaErrorString(e);
      goto cleanup;
    }
    e = cudaStreamCreate(&streams[i]);
    if (e != cudaSuccess) {
      res.error = "cudaStreamCreate failed on GPU " + std::to_string(i) + ": " + cudaErrorString(e);
      goto cleanup;
    }
  }

  {
    int r = nccl.ncclCommInitAll(comms.data(), deviceCount, devs.data());
    if (r != 0) {
      res.error = "ncclCommInitAll failed: " + nccl.err(r);
      goto cleanup;
    }
  }

  res.bytesPerRank = opt.ncclBytes;
  if (!allocateNcclBuffers(devs, res.bytesPerRank, send, recv, res.error)) {
    goto cleanup;
  }

  {
    const size_t countFloats = res.bytesPerRank / sizeof(float);
    const int warmupIters = 4;
    if (!launchAllReduceBatch(nccl, devs, comms, send, recv, streams, countFloats,
                              warmupIters, res.error)) {
      goto cleanup;
    }

    const auto start = Clock::now();
    int iters = 0;
    double elapsed = 0.0;
    do {
      if (!launchAllReduceBatch(nccl, devs, comms, send, recv, streams, countFloats,
                                opt.ncclBatch, res.error)) {
        goto cleanup;
      }
      iters += opt.ncclBatch;
      elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    } while (elapsed < opt.effectiveAllSeconds());

    res.iterations = iters;
    res.elapsedSec = elapsed;
    if (elapsed > 0.0 && iters > 0) {
      res.algGBps = (double)res.bytesPerRank * (double)iters / elapsed / 1e9;
      res.busGBps = res.algGBps * (2.0 * (double)(deviceCount - 1) / (double)deviceCount);
      res.ran = true;
    }
  }

cleanup:
  for (int i = 0; i < deviceCount; ++i) {
    if (streams[i]) {
      cudaSetDevice(devs[i]);
      cudaStreamDestroy(streams[i]);
    }
  }
  freeDeviceBuffers(devs, send, recv);
  destroyNcclComms(nccl, comms);
  return res;
}


struct PairMethodResult {
  bool ran = false;
  int iterations = 0;
  double elapsedSec = 0.0;
  double gbps = 0.0;
  std::string error;
};

struct PeerAccessInfo {
  bool srcCanAccessDst = false;  // Kernel on src can directly access memory allocated on dst.
  bool dstCanAccessSrc = false;  // Kernel on dst can directly access memory allocated on src.
  bool srcToDstEnabled = false;
  bool dstToSrcEnabled = false;
  std::string error;
};

struct GpuPairResult {
  int src = -1;
  int dst = -1;
  size_t bytes = 0;
  std::string linkToken = "UNKNOWN";
  std::string linkDetail = "UNKNOWN";
  PeerAccessInfo peer;
  PairMethodResult cudaCopy;
  PairMethodResult cudaCopyBiDir;  // Simultaneous src->dst and dst->src. Reported as total payload GB/s.
  PairMethodResult peerRead;   // Kernel on dst reads memory from src.
  PairMethodResult peerWrite;  // Kernel on src writes memory to dst.
  PairMethodResult hostStaged;
  PairMethodResult ncclSendRecv;
  double bestGBps = 0.0;
  std::string bestMethod;
  std::string bestPath;
  std::string error;
};

static std::string speedToString(double gbps) {
  if (!(gbps > 0.0)) return "n/a";
  std::ostringstream os;
  os << std::fixed << std::setprecision(2) << gbps;
  return os.str();
}

static void appendError(std::string& dst, const std::string& msg) {
  if (msg.empty()) return;
  if (!dst.empty()) dst += "; ";
  dst += msg;
}


static std::string trimCopy(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

static std::vector<std::string> splitWhitespace(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream is(s);
  std::string tok;
  while (is >> tok) out.push_back(tok);
  return out;
}

static bool parseGpuLabelIndex(const std::string& token, int& index) {
  if (token.size() < 4) return false;
  if (token[0] != 'G' || token[1] != 'P' || token[2] != 'U') return false;
  for (size_t i = 3; i < token.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(token[i]))) return false;
  }
  try {
    index = std::stoi(token.substr(3));
    return true;
  } catch (...) {
    return false;
  }
}

static std::string captureCommandOutput(const char* cmd) {
  std::string out;
  FILE* pipe = popen(cmd, "r");
  if (!pipe) return out;
  char buf[4096];
  while (fgets(buf, sizeof(buf), pipe)) out += buf;
  pclose(pipe);
  return out;
}

static std::string fitColumn(const std::string& s, size_t width) {
  if (s.size() <= width) return s;
  if (width <= 3) return s.substr(0, width);
  return s.substr(0, width - 3) + "...";
}

struct PciBusIdParts {
  bool valid = false;
  unsigned int domain = 0;
  unsigned int bus = 0;
  unsigned int device = 0;
  unsigned int function = 0;
};

static bool parsePciBusIdParts(const std::string& raw, PciBusIdParts& out) {
  std::string s = trimCopy(raw);
  unsigned int domain = 0, bus = 0, device = 0, function = 0;
  if (std::sscanf(s.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) == 4) {
    out.valid = true;
    out.domain = domain;
    out.bus = bus;
    out.device = device;
    out.function = function;
    return true;
  }
  if (std::sscanf(s.c_str(), "%x:%x.%x", &bus, &device, &function) == 3) {
    out.valid = true;
    out.domain = 0;
    out.bus = bus;
    out.device = device;
    out.function = function;
    return true;
  }
  return false;
}

static bool samePciBusId(const PciBusIdParts& a, const PciBusIdParts& b) {
  return a.valid && b.valid && a.domain == b.domain && a.bus == b.bus &&
         a.device == b.device && a.function == b.function;
}

struct GpuLinkTopology {
  bool available = false;
  std::string source = "nvidia-smi topo -m";
  std::string error;
  std::vector<int> cudaToNvidiaIndex;
  std::vector<std::vector<std::string>> link;

  std::string linkFor(int src, int dst) const {
    if (src < 0 || dst < 0 || src >= (int)link.size() || dst >= (int)link[(size_t)src].size()) {
      return "UNKNOWN";
    }
    const std::string& v = link[(size_t)src][(size_t)dst];
    return v.empty() ? "UNKNOWN" : v;
  }
};

static int fillCudaP2PFallbackLinks(GpuLinkTopology& topo, int deviceCount) {
  int filled = 0;
  if ((int)topo.link.size() != deviceCount) {
    topo.link.assign((size_t)deviceCount, std::vector<std::string>((size_t)deviceCount, "UNKNOWN"));
  }
  for (int src = 0; src < deviceCount; ++src) {
    for (int dst = 0; dst < deviceCount; ++dst) {
      if (src == dst) {
        topo.link[(size_t)src][(size_t)dst] = "X";
        continue;
      }
      if (topo.linkFor(src, dst) != "UNKNOWN") continue;
      int can = 0;
      cudaError_t e = cudaDeviceCanAccessPeer(&can, src, dst);
      if (e == cudaSuccess) {
        topo.link[(size_t)src][(size_t)dst] = can ? "CUDA_P2P" : "HOST/PCIe";
        ++filled;
      } else {
        cudaGetLastError();
      }
    }
  }
  return filled;
}

static std::vector<std::pair<int, PciBusIdParts>> queryNvidiaSmiGpuPciMap() {
  std::vector<std::pair<int, PciBusIdParts>> result;
  const std::string out = captureCommandOutput(
      "nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits 2>/dev/null");
  std::istringstream is(out);
  std::string line;
  while (std::getline(is, line)) {
    line = trimCopy(line);
    if (line.empty()) continue;
    const size_t comma = line.find(',');
    if (comma == std::string::npos) continue;
    const std::string idxText = trimCopy(line.substr(0, comma));
    const std::string pciText = trimCopy(line.substr(comma + 1));
    int idx = -1;
    try {
      idx = std::stoi(idxText);
    } catch (...) {
      continue;
    }
    PciBusIdParts pci;
    if (parsePciBusIdParts(pciText, pci)) result.push_back({idx, pci});
  }
  return result;
}

static std::string describeTopologyToken(const std::string& token) {
  if (token.empty() || token == "UNKNOWN") return "UNKNOWN";
  if (token == "X") return "self";
  if (token.size() >= 2 && token[0] == 'N' && token[1] == 'V') {
    std::string n = token.substr(2);
    if (!n.empty()) return "NVLink/NVSwitch, bonded " + n + " NVLinks";
    return "NVLink/NVSwitch";
  }
  if (token == "PIX") return "PCIe, at most one PCIe switch";
  if (token == "PXB") return "PCIe, multiple PCIe switches";
  if (token == "PHB") return "PCIe through host bridge/CPU root complex";
  if (token == "NODE") return "PCIe plus host-bridge path inside one NUMA node";
  if (token == "SYS") return "PCIe plus cross-NUMA SMP interconnect";
  if (token == "SOC") return "system-on-chip/on-package interconnect";
  if (token == "PXN") return "PCIe/NVLink path through an intermediate GPU/NIC topology";
  if (token == "CUDA_P2P") return "CUDA P2P-capable path; exact physical topology unavailable";
  if (token == "HOST/PCIe") return "No CUDA P2P direct access; likely host/PCIe staged path";
  return token;
}

static GpuLinkTopology buildGpuLinkTopology(int deviceCount) {
  GpuLinkTopology topo;
  topo.cudaToNvidiaIndex.assign((size_t)deviceCount, -1);
  topo.link.assign((size_t)deviceCount, std::vector<std::string>((size_t)deviceCount, "UNKNOWN"));
  for (int i = 0; i < deviceCount; ++i) topo.link[(size_t)i][(size_t)i] = "X";

  const std::string topoOut = captureCommandOutput("nvidia-smi topo -m 2>/dev/null");
  if (topoOut.empty()) {
    const int filled = fillCudaP2PFallbackLinks(topo, deviceCount);
    topo.available = (filled > 0);
    topo.source = "CUDA cudaDeviceCanAccessPeer fallback";
    topo.error = "nvidia-smi topo -m produced no output";
    return topo;
  }

  std::vector<int> headerGpuIndices;
  std::vector<int> headerTokenPositions;
  std::map<int, std::map<int, std::string>> nvTopo;
  bool headerFound = false;

  std::istringstream lines(topoOut);
  std::string line;
  while (std::getline(lines, line)) {
    std::vector<std::string> toks = splitWhitespace(line);
    if (toks.empty()) continue;

    if (!headerFound) {
      std::vector<int> candIdx;
      std::vector<int> candPos;
      for (int pos = 0; pos < (int)toks.size(); ++pos) {
        int gpuIndex = -1;
        if (parseGpuLabelIndex(toks[(size_t)pos], gpuIndex)) {
          candIdx.push_back(gpuIndex);
          candPos.push_back(pos);
        }
      }
      if ((int)candIdx.size() >= 2) {
        headerGpuIndices = candIdx;
        headerTokenPositions = candPos;
        headerFound = true;
      }
      continue;
    }

    if (toks[0] == "Legend:") break;
    int rowGpu = -1;
    if (!parseGpuLabelIndex(toks[0], rowGpu)) continue;
    for (size_t k = 0; k < headerGpuIndices.size(); ++k) {
      const int colGpu = headerGpuIndices[k];
      const int rowValuePos = 1 + headerTokenPositions[k];
      if (rowValuePos >= 0 && rowValuePos < (int)toks.size()) {
        nvTopo[rowGpu][colGpu] = toks[(size_t)rowValuePos];
      }
    }
  }

  if (nvTopo.empty()) {
    const int filled = fillCudaP2PFallbackLinks(topo, deviceCount);
    topo.available = (filled > 0);
    topo.source = "CUDA cudaDeviceCanAccessPeer fallback";
    topo.error = "could not parse GPU rows from nvidia-smi topo -m";
    return topo;
  }

  const std::vector<std::pair<int, PciBusIdParts>> nvsmiPci = queryNvidiaSmiGpuPciMap();
  for (int cudaDev = 0; cudaDev < deviceCount; ++cudaDev) {
    char pciText[64] = {0};
    cudaError_t ce = cudaDeviceGetPCIBusId(pciText, sizeof(pciText), cudaDev);
    if (ce != cudaSuccess) continue;
    PciBusIdParts cudaPci;
    if (!parsePciBusIdParts(pciText, cudaPci)) continue;
    for (const auto& item : nvsmiPci) {
      if (samePciBusId(cudaPci, item.second)) {
        topo.cudaToNvidiaIndex[(size_t)cudaDev] = item.first;
        break;
      }
    }
  }

  // Fallback for the common case where CUDA ordinals and nvidia-smi GPU indexes match.
  for (int cudaDev = 0; cudaDev < deviceCount; ++cudaDev) {
    if (topo.cudaToNvidiaIndex[(size_t)cudaDev] < 0 && nvTopo.count(cudaDev)) {
      topo.cudaToNvidiaIndex[(size_t)cudaDev] = cudaDev;
    }
  }

  bool anyKnown = false;
  for (int src = 0; src < deviceCount; ++src) {
    for (int dst = 0; dst < deviceCount; ++dst) {
      if (src == dst) {
        topo.link[(size_t)src][(size_t)dst] = "X";
        continue;
      }
      const int nvSrc = topo.cudaToNvidiaIndex[(size_t)src];
      const int nvDst = topo.cudaToNvidiaIndex[(size_t)dst];
      auto rowIt = nvTopo.find(nvSrc);
      if (rowIt != nvTopo.end()) {
        auto colIt = rowIt->second.find(nvDst);
        if (colIt != rowIt->second.end() && !colIt->second.empty()) {
          topo.link[(size_t)src][(size_t)dst] = colIt->second;
          anyKnown = true;
        }
      }
    }
  }

  const int fallbackFilled = fillCudaP2PFallbackLinks(topo, deviceCount);
  topo.available = anyKnown || fallbackFilled > 0;
  if (!anyKnown && fallbackFilled > 0) {
    topo.source = "CUDA cudaDeviceCanAccessPeer fallback";
    topo.error = "topology parsed, but CUDA GPU PCI IDs could not be mapped to nvidia-smi GPU indexes";
  } else if (anyKnown && fallbackFilled > 0) {
    topo.source = "nvidia-smi topo -m plus CUDA P2P fallback for unknown pairs";
  } else if (!anyKnown) {
    topo.error = "topology parsed, but CUDA GPU PCI IDs could not be mapped to nvidia-smi GPU indexes";
  }
  return topo;
}

static std::string summarizeLinksForGpu(const GpuLinkTopology& topo, int gpu, int deviceCount) {
  std::map<std::string, int> counts;
  for (int peer = 0; peer < deviceCount; ++peer) {
    if (peer == gpu) continue;
    counts[topo.linkFor(gpu, peer)]++;
  }
  std::ostringstream os;
  bool first = true;
  for (const auto& kv : counts) {
    if (!first) os << ",";
    first = false;
    os << kv.first << "x" << kv.second;
  }
  return os.str().empty() ? "UNKNOWN" : os.str();
}

static void printGpuLinkTopologyMatrix(const GpuLinkTopology& topo, int deviceCount) {
  if (!topo.available) {
    std::cout << "GPU-to-GPU topology link detection: unavailable (" << topo.error << ").\n";
    return;
  }

  std::cout << "\nGPU-to-GPU topology links from " << topo.source << ":\n";
  std::cout << std::setw(8) << "src\\dst";
  for (int dst = 0; dst < deviceCount; ++dst) {
    std::ostringstream label;
    label << "GPU" << dst;
    std::cout << std::setw(10) << label.str();
  }
  std::cout << "\n";
  for (int src = 0; src < deviceCount; ++src) {
    std::ostringstream label;
    label << "GPU" << src;
    std::cout << std::setw(8) << label.str();
    for (int dst = 0; dst < deviceCount; ++dst) {
      std::cout << std::setw(10) << topo.linkFor(src, dst);
    }
    std::cout << "\n";
  }
  std::cout << "Link legend: NV#=NVLink/NVSwitch bonded NVLinks, PIX/PXB/PHB/NODE/SYS=PCIe paths with increasing distance.\n";
}

static bool queryAndEnablePeerAccessDirection(int activeDev, int peerDev,
                                              bool& canAccess, bool& enabled,
                                              std::string& error) {
  canAccess = false;
  enabled = false;

  cudaError_t e = cudaSetDevice(activeDev);
  if (e != cudaSuccess) {
    error = "cudaSetDevice failed on GPU " + std::to_string(activeDev) + ": " + cudaErrorString(e);
    return false;
  }

  int can = 0;
  e = cudaDeviceCanAccessPeer(&can, activeDev, peerDev);
  if (e != cudaSuccess) {
    error = "cudaDeviceCanAccessPeer(" + std::to_string(activeDev) + "," +
            std::to_string(peerDev) + ") failed: " + cudaErrorString(e);
    return false;
  }
  canAccess = (can != 0);
  if (!canAccess) return true;

  e = cudaDeviceEnablePeerAccess(peerDev, 0);
  if (e == cudaSuccess) {
    enabled = true;
    return true;
  }
  if (e == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
    enabled = true;
    return true;
  }

  error = "cudaDeviceEnablePeerAccess(active=" + std::to_string(activeDev) +
          ", peer=" + std::to_string(peerDev) + ") failed: " + cudaErrorString(e);
  cudaGetLastError();
  return false;
}

static PeerAccessInfo setupPeerAccessForPair(int src, int dst) {
  PeerAccessInfo p;
  std::string err;
  if (!queryAndEnablePeerAccessDirection(src, dst, p.srcCanAccessDst, p.srcToDstEnabled, err)) {
    appendError(p.error, err);
  }
  err.clear();
  if (!queryAndEnablePeerAccessDirection(dst, src, p.dstCanAccessSrc, p.dstToSrcEnabled, err)) {
    appendError(p.error, err);
  }
  return p;
}

static void disablePeerAccessDirection(int activeDev, int peerDev, bool enabled) {
  if (!enabled) return;
  cudaError_t e = cudaSetDevice(activeDev);
  if (e != cudaSuccess) return;
  e = cudaDeviceDisablePeerAccess(peerDev);
  if (e != cudaSuccess) cudaGetLastError();
}

static void disablePeerAccessForPair(int src, int dst, const PeerAccessInfo& p) {
  disablePeerAccessDirection(src, dst, p.srcToDstEnabled);
  disablePeerAccessDirection(dst, src, p.dstToSrcEnabled);
}

static void freePairBuffers(int src, int dst, void*& srcBuf, void*& dstBuf) {
  if (srcBuf) {
    cudaSetDevice(src);
    cudaFree(srcBuf);
    srcBuf = nullptr;
  }
  if (dstBuf) {
    cudaSetDevice(dst);
    cudaFree(dstBuf);
    dstBuf = nullptr;
  }
}

static bool allocateG2GBuffers(int src, int dst, size_t& bytes,
                               void*& srcBuf, void*& dstBuf,
                               std::string& error) {
  srcBuf = nullptr;
  dstBuf = nullptr;
  bytes = (bytes / sizeof(uint4)) * sizeof(uint4);
  if (bytes < MiB) bytes = MiB;

  while (bytes >= MiB) {
    error.clear();
    bool ok = true;

    cudaError_t e = cudaSetDevice(src);
    if (e != cudaSuccess) {
      error = "cudaSetDevice failed on source GPU " + std::to_string(src) + ": " + cudaErrorString(e);
      ok = false;
    }
    if (ok) {
      e = cudaMalloc(&srcBuf, bytes);
      if (e != cudaSuccess) {
        cudaGetLastError();
        error = "cudaMalloc source buffer failed on GPU " + std::to_string(src) + ": " + cudaErrorString(e);
        ok = false;
      }
    }
    if (ok) {
      e = cudaSetDevice(dst);
      if (e != cudaSuccess) {
        error = "cudaSetDevice failed on destination GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
        ok = false;
      }
    }
    if (ok) {
      e = cudaMalloc(&dstBuf, bytes);
      if (e != cudaSuccess) {
        cudaGetLastError();
        error = "cudaMalloc destination buffer failed on GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
        ok = false;
      }
    }
    if (ok) {
      cudaSetDevice(src);
      e = cudaMemset(srcBuf, 0x5A, bytes);
      if (e == cudaSuccess) e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        error = "cudaMemset/sync source buffer failed on GPU " + std::to_string(src) + ": " + cudaErrorString(e);
        ok = false;
      }
    }
    if (ok) {
      cudaSetDevice(dst);
      e = cudaMemset(dstBuf, 0x00, bytes);
      if (e == cudaSuccess) e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        error = "cudaMemset/sync destination buffer failed on GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
        ok = false;
      }
    }

    if (ok) return true;
    freePairBuffers(src, dst, srcBuf, dstBuf);
    bytes /= 2;
    bytes = (bytes / sizeof(uint4)) * sizeof(uint4);
  }

  if (error.empty()) error = "GPU-to-GPU buffer allocation failed; buffer dropped below 1 MiB";
  return false;
}

static bool createPairStreams(int src, int dst, cudaStream_t& srcStream,
                              cudaStream_t& dstStream, std::string& error) {
  srcStream = nullptr;
  dstStream = nullptr;

  cudaError_t e = cudaSetDevice(src);
  if (e != cudaSuccess) {
    error = "cudaSetDevice failed on source GPU " + std::to_string(src) + ": " + cudaErrorString(e);
    return false;
  }
  e = cudaStreamCreate(&srcStream);
  if (e != cudaSuccess) {
    error = "cudaStreamCreate failed on source GPU " + std::to_string(src) + ": " + cudaErrorString(e);
    return false;
  }

  e = cudaSetDevice(dst);
  if (e != cudaSuccess) {
    error = "cudaSetDevice failed on destination GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    return false;
  }
  e = cudaStreamCreate(&dstStream);
  if (e != cudaSuccess) {
    error = "cudaStreamCreate failed on destination GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    return false;
  }
  return true;
}

static void destroyPairStreams(int src, int dst, cudaStream_t& srcStream, cudaStream_t& dstStream) {
  if (srcStream) {
    cudaSetDevice(src);
    cudaStreamDestroy(srcStream);
    srcStream = nullptr;
  }
  if (dstStream) {
    cudaSetDevice(dst);
    cudaStreamDestroy(dstStream);
    dstStream = nullptr;
  }
}

static bool syncPairStreams(int src, int dst, cudaStream_t srcStream, cudaStream_t dstStream,
                            std::string& error) {
  cudaError_t e = cudaSetDevice(src);
  if (e != cudaSuccess) {
    error = "cudaSetDevice failed on source GPU " + std::to_string(src) + ": " + cudaErrorString(e);
    return false;
  }
  e = cudaStreamSynchronize(srcStream);
  if (e != cudaSuccess) {
    error = "cudaStreamSynchronize failed on source GPU " + std::to_string(src) + ": " + cudaErrorString(e);
    return false;
  }

  e = cudaSetDevice(dst);
  if (e != cudaSuccess) {
    error = "cudaSetDevice failed on destination GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    return false;
  }
  e = cudaStreamSynchronize(dstStream);
  if (e != cudaSuccess) {
    error = "cudaStreamSynchronize failed on destination GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    return false;
  }
  return true;
}

static PairMethodResult runCudaMemcpyPeerBandwidth(int src, int dst,
                                                   void* srcBuf, void* dstBuf,
                                                   size_t bytes, double seconds,
                                                   int batch, cudaStream_t dstStream) {
  PairMethodResult m;
  if (batch <= 0) batch = 1;

  cudaError_t e = cudaSetDevice(dst);
  if (e != cudaSuccess) {
    m.error = "cudaSetDevice failed on destination GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    return m;
  }

  e = cudaMemcpyPeerAsync(dstBuf, dst, srcBuf, src, bytes, dstStream);
  if (e == cudaSuccess) e = cudaStreamSynchronize(dstStream);
  if (e != cudaSuccess) {
    m.error = "cudaMemcpyPeerAsync warmup failed from GPU " + std::to_string(src) +
              " to GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    cudaGetLastError();
    return m;
  }

  const auto start = Clock::now();
  int iters = 0;
  double elapsed = 0.0;
  do {
    for (int b = 0; b < batch; ++b) {
      e = cudaMemcpyPeerAsync(dstBuf, dst, srcBuf, src, bytes, dstStream);
      if (e != cudaSuccess) {
        m.error = "cudaMemcpyPeerAsync failed from GPU " + std::to_string(src) +
                  " to GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
        cudaGetLastError();
        return m;
      }
    }
    e = cudaStreamSynchronize(dstStream);
    if (e != cudaSuccess) {
      m.error = "cudaMemcpyPeerAsync sync failed from GPU " + std::to_string(src) +
                " to GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
      return m;
    }
    iters += batch;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < seconds);

  m.ran = true;
  m.iterations = iters;
  m.elapsedSec = elapsed;
  if (elapsed > 0.0 && iters > 0) {
    m.gbps = (double)bytes * (double)iters / elapsed / 1e9;
  }
  return m;
}


struct ConcurrentCopyTask {
  int src = -1;
  int dst = -1;
  void* srcPtr = nullptr;
  void* dstPtr = nullptr;
  cudaStream_t stream = nullptr;  // Created on dst device.
};

static bool createConcurrentCopyStreams(std::vector<ConcurrentCopyTask>& tasks, std::string& error) {
  for (auto& t : tasks) {
    cudaError_t e = cudaSetDevice(t.dst);
    if (e != cudaSuccess) {
      error = "cudaSetDevice failed while creating copy stream on GPU " + std::to_string(t.dst) +
              ": " + cudaErrorString(e);
      return false;
    }
    e = cudaStreamCreate(&t.stream);
    if (e != cudaSuccess) {
      error = "cudaStreamCreate failed for GPU-to-GPU aggregate copy on GPU " +
              std::to_string(t.dst) + ": " + cudaErrorString(e);
      return false;
    }
  }
  return true;
}

static void destroyConcurrentCopyStreams(std::vector<ConcurrentCopyTask>& tasks) {
  for (auto& t : tasks) {
    if (!t.stream) continue;
    cudaSetDevice(t.dst);
    cudaStreamDestroy(t.stream);
    t.stream = nullptr;
  }
}

static bool enqueueConcurrentPeerCopies(const std::vector<ConcurrentCopyTask>& tasks,
                                        size_t bytes, int repeats,
                                        std::string& error) {
  if (repeats <= 0) repeats = 1;
  for (int r = 0; r < repeats; ++r) {
    for (const auto& t : tasks) {
      cudaError_t e = cudaSetDevice(t.dst);
      if (e != cudaSuccess) {
        error = "cudaSetDevice failed while enqueueing copy " + std::to_string(t.src) +
                " -> " + std::to_string(t.dst) + ": " + cudaErrorString(e);
        return false;
      }
      e = cudaMemcpyPeerAsync(t.dstPtr, t.dst, t.srcPtr, t.src, bytes, t.stream);
      if (e != cudaSuccess) {
        error = "cudaMemcpyPeerAsync failed while enqueueing copy " + std::to_string(t.src) +
                " -> " + std::to_string(t.dst) + ": " + cudaErrorString(e);
        cudaGetLastError();
        return false;
      }
    }
  }
  return true;
}

static bool syncConcurrentCopyStreams(const std::vector<ConcurrentCopyTask>& tasks,
                                      std::string& error) {
  for (const auto& t : tasks) {
    cudaError_t e = cudaSetDevice(t.dst);
    if (e != cudaSuccess) {
      error = "cudaSetDevice failed while synchronizing copy stream on GPU " +
              std::to_string(t.dst) + ": " + cudaErrorString(e);
      return false;
    }
    e = cudaStreamSynchronize(t.stream);
    if (e != cudaSuccess) {
      error = "cudaStreamSynchronize failed for copy " + std::to_string(t.src) +
              " -> " + std::to_string(t.dst) + ": " + cudaErrorString(e);
      return false;
    }
  }
  return true;
}

static PairMethodResult runConcurrentCudaMemcpyPeerBandwidth(
    const std::vector<ConcurrentCopyTask>& tasks, size_t bytes, double seconds,
    int batch, const std::string& label) {
  PairMethodResult m;
  if (tasks.empty()) {
    m.error = label + ": no copy tasks";
    return m;
  }
  if (batch <= 0) batch = 1;

  if (!enqueueConcurrentPeerCopies(tasks, bytes, 2, m.error) ||
      !syncConcurrentCopyStreams(tasks, m.error)) {
    if (m.error.empty()) m.error = label + ": warmup failed";
    else m.error = label + ": warmup failed: " + m.error;
    return m;
  }

  const auto start = Clock::now();
  int iters = 0;
  double elapsed = 0.0;
  do {
    if (!enqueueConcurrentPeerCopies(tasks, bytes, batch, m.error) ||
        !syncConcurrentCopyStreams(tasks, m.error)) {
      if (m.error.empty()) m.error = label + ": timed copy failed";
      else m.error = label + ": timed copy failed: " + m.error;
      return m;
    }
    iters += batch;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < seconds);

  m.ran = true;
  m.iterations = iters;
  m.elapsedSec = elapsed;
  if (elapsed > 0.0 && iters > 0) {
    m.gbps = (double)bytes * (double)tasks.size() * (double)iters / elapsed / 1e9;
  }
  return m;
}

static PairMethodResult runPeerRemoteReadBandwidth(int src, int dst,
                                                   const std::vector<cudaDeviceProp>& props,
                                                   void* srcBuf, void* dstBuf,
                                                   size_t bytes, double seconds,
                                                   int batch, cudaStream_t dstStream) {
  PairMethodResult m;
  if (batch <= 0) batch = 1;
  const size_t elems = bytes / sizeof(uint4);
  if (elems == 0) {
    m.error = "buffer too small for uint4 peer read test";
    return m;
  }

  cudaError_t e = cudaSetDevice(dst);
  if (e != cudaSuccess) {
    m.error = "cudaSetDevice failed on destination GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    return m;
  }

  const int threads = 256;
  const int blocks = std::max(1, std::min(props[dst].multiProcessorCount * 32, 65535));

  peerRemoteReadKernel<<<blocks, threads, 0, dstStream>>>(
      reinterpret_cast<const uint4*>(srcBuf), reinterpret_cast<uint4*>(dstBuf), elems, 0x12345678u);
  e = cudaPeekAtLastError();
  if (e == cudaSuccess) e = cudaStreamSynchronize(dstStream);
  if (e != cudaSuccess) {
    m.error = "peer remote-read warmup failed, GPU " + std::to_string(dst) +
              " reading GPU " + std::to_string(src) + ": " + cudaErrorString(e);
    cudaGetLastError();
    return m;
  }

  const auto start = Clock::now();
  int iters = 0;
  double elapsed = 0.0;
  do {
    for (int b = 0; b < batch; ++b) {
      peerRemoteReadKernel<<<blocks, threads, 0, dstStream>>>(
          reinterpret_cast<const uint4*>(srcBuf), reinterpret_cast<uint4*>(dstBuf), elems,
          0x9e3779b9u + (unsigned int)(iters + b));
      e = cudaPeekAtLastError();
      if (e != cudaSuccess) {
        m.error = "peer remote-read kernel launch failed, GPU " + std::to_string(dst) +
                  " reading GPU " + std::to_string(src) + ": " + cudaErrorString(e);
        cudaGetLastError();
        return m;
      }
    }
    e = cudaStreamSynchronize(dstStream);
    if (e != cudaSuccess) {
      m.error = "peer remote-read kernel execution failed, GPU " + std::to_string(dst) +
                " reading GPU " + std::to_string(src) + ": " + cudaErrorString(e);
      return m;
    }
    iters += batch;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < seconds);

  m.ran = true;
  m.iterations = iters;
  m.elapsedSec = elapsed;
  if (elapsed > 0.0 && iters > 0) {
    m.gbps = (double)bytes * (double)iters / elapsed / 1e9;
  }
  return m;
}

static PairMethodResult runPeerRemoteWriteBandwidth(int src, int dst,
                                                    const std::vector<cudaDeviceProp>& props,
                                                    void* srcBuf, void* dstBuf,
                                                    size_t bytes, double seconds,
                                                    int batch, cudaStream_t srcStream) {
  PairMethodResult m;
  if (batch <= 0) batch = 1;
  const size_t elems = bytes / sizeof(uint4);
  if (elems == 0) {
    m.error = "buffer too small for uint4 peer write test";
    return m;
  }

  cudaError_t e = cudaSetDevice(src);
  if (e != cudaSuccess) {
    m.error = "cudaSetDevice failed on source GPU " + std::to_string(src) + ": " + cudaErrorString(e);
    return m;
  }

  const int threads = 256;
  const int blocks = std::max(1, std::min(props[src].multiProcessorCount * 32, 65535));

  peerRemoteWriteKernel<<<blocks, threads, 0, srcStream>>>(
      reinterpret_cast<const uint4*>(srcBuf), reinterpret_cast<uint4*>(dstBuf), elems, 0x12345678u);
  e = cudaPeekAtLastError();
  if (e == cudaSuccess) e = cudaStreamSynchronize(srcStream);
  if (e != cudaSuccess) {
    m.error = "peer remote-write warmup failed, GPU " + std::to_string(src) +
              " writing GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    cudaGetLastError();
    return m;
  }

  const auto start = Clock::now();
  int iters = 0;
  double elapsed = 0.0;
  do {
    for (int b = 0; b < batch; ++b) {
      peerRemoteWriteKernel<<<blocks, threads, 0, srcStream>>>(
          reinterpret_cast<const uint4*>(srcBuf), reinterpret_cast<uint4*>(dstBuf), elems,
          0x85ebca6bu + (unsigned int)(iters + b));
      e = cudaPeekAtLastError();
      if (e != cudaSuccess) {
        m.error = "peer remote-write kernel launch failed, GPU " + std::to_string(src) +
                  " writing GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
        cudaGetLastError();
        return m;
      }
    }
    e = cudaStreamSynchronize(srcStream);
    if (e != cudaSuccess) {
      m.error = "peer remote-write kernel execution failed, GPU " + std::to_string(src) +
                " writing GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
      return m;
    }
    iters += batch;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < seconds);

  m.ran = true;
  m.iterations = iters;
  m.elapsedSec = elapsed;
  if (elapsed > 0.0 && iters > 0) {
    m.gbps = (double)bytes * (double)iters / elapsed / 1e9;
  }
  return m;
}

static PairMethodResult runHostStagedBandwidth(int src, int dst,
                                               void* srcBuf, void* dstBuf, void* hostBuf,
                                               size_t bytes, double seconds, int batch,
                                               cudaStream_t srcStream, cudaStream_t dstStream) {
  PairMethodResult m;
  if (batch <= 0) batch = 1;
  if (!hostBuf) {
    m.error = "no pinned host buffer available";
    return m;
  }

  auto copyOnce = [&]() -> cudaError_t {
    cudaError_t e = cudaSetDevice(src);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(hostBuf, srcBuf, bytes, cudaMemcpyDeviceToHost, srcStream);
    if (e != cudaSuccess) return e;
    e = cudaStreamSynchronize(srcStream);
    if (e != cudaSuccess) return e;

    e = cudaSetDevice(dst);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(dstBuf, hostBuf, bytes, cudaMemcpyHostToDevice, dstStream);
    if (e != cudaSuccess) return e;
    e = cudaStreamSynchronize(dstStream);
    return e;
  };

  cudaError_t e = copyOnce();
  if (e != cudaSuccess) {
    m.error = "host-staged D2H/H2D warmup failed from GPU " + std::to_string(src) +
              " to GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
    cudaGetLastError();
    return m;
  }

  const auto start = Clock::now();
  int iters = 0;
  double elapsed = 0.0;
  do {
    for (int b = 0; b < batch; ++b) {
      e = copyOnce();
      if (e != cudaSuccess) {
        m.error = "host-staged D2H/H2D copy failed from GPU " + std::to_string(src) +
                  " to GPU " + std::to_string(dst) + ": " + cudaErrorString(e);
        cudaGetLastError();
        return m;
      }
    }
    iters += batch;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < seconds);

  m.ran = true;
  m.iterations = iters;
  m.elapsedSec = elapsed;
  if (elapsed > 0.0 && iters > 0) {
    // Logical GPU-to-GPU payload bandwidth. Actual PCIe traffic is roughly 2x because data is staged through host memory.
    m.gbps = (double)bytes * (double)iters / elapsed / 1e9;
  }
  return m;
}

static bool launchNcclSendRecvBatch(NcclApi& nccl,
                                    const std::vector<ncclComm_t_dyn>& comms,
                                    int src, int dst, void* srcBuf, void* dstBuf,
                                    size_t countFloats, int batch,
                                    cudaStream_t srcStream, cudaStream_t dstStream,
                                    std::string& error) {
  int r = nccl.ncclGroupStart();
  if (r != 0) {
    error = "ncclGroupStart failed: " + nccl.err(r);
    return false;
  }

  for (int b = 0; b < batch; ++b) {
    cudaSetDevice(src);
    r = nccl.ncclSend(srcBuf, countFloats, NCCL_DTYPE_FLOAT32, dst, comms[src], srcStream);
    if (r != 0) {
      error = "ncclSend enqueue failed from GPU " + std::to_string(src) +
              " to GPU " + std::to_string(dst) + ": " + nccl.err(r);
      nccl.ncclGroupEnd();
      return false;
    }

    cudaSetDevice(dst);
    r = nccl.ncclRecv(dstBuf, countFloats, NCCL_DTYPE_FLOAT32, src, comms[dst], dstStream);
    if (r != 0) {
      error = "ncclRecv enqueue failed on GPU " + std::to_string(dst) +
              " from GPU " + std::to_string(src) + ": " + nccl.err(r);
      nccl.ncclGroupEnd();
      return false;
    }
  }

  r = nccl.ncclGroupEnd();
  if (r != 0) {
    error = "ncclGroupEnd failed: " + nccl.err(r);
    return false;
  }

  return syncPairStreams(src, dst, srcStream, dstStream, error);
}

static PairMethodResult runNcclSendRecvBandwidth(NcclApi& nccl,
                                                 const std::vector<ncclComm_t_dyn>& comms,
                                                 int src, int dst,
                                                 void* srcBuf, void* dstBuf,
                                                 size_t bytes, double seconds, int batch,
                                                 cudaStream_t srcStream, cudaStream_t dstStream) {
  PairMethodResult m;
  if (!nccl.hasSendRecv()) {
    m.error = "NCCL ncclSend/ncclRecv symbols are not available";
    return m;
  }
  if (src < 0 || dst < 0 || src >= (int)comms.size() || dst >= (int)comms.size() ||
      !comms[src] || !comms[dst]) {
    m.error = "NCCL communicators are not ready for this pair";
    return m;
  }
  if (batch <= 0) batch = 1;

  const size_t countFloats = bytes / sizeof(float);
  if (countFloats == 0) {
    m.error = "buffer too small for NCCL Send/Recv test";
    return m;
  }

  if (!launchNcclSendRecvBatch(nccl, comms, src, dst, srcBuf, dstBuf,
                               countFloats, 2, srcStream, dstStream, m.error)) {
    return m;
  }

  const auto start = Clock::now();
  int iters = 0;
  double elapsed = 0.0;
  do {
    if (!launchNcclSendRecvBatch(nccl, comms, src, dst, srcBuf, dstBuf,
                                 countFloats, batch, srcStream, dstStream, m.error)) {
      return m;
    }
    iters += batch;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < seconds);

  m.ran = true;
  m.iterations = iters;
  m.elapsedSec = elapsed;
  if (elapsed > 0.0 && iters > 0) {
    m.gbps = (double)bytes * (double)iters / elapsed / 1e9;
  }
  return m;
}

static GpuPairResult runGpuToGpuPairBandwidth(int src, int dst,
                                              const std::vector<cudaDeviceProp>& props,
                                              NcclApi* nccl,
                                              const std::vector<ncclComm_t_dyn>& ncclComms,
                                              const Options& opt,
                                              const GpuLinkTopology* topology) {
  GpuPairResult r;
  r.src = src;
  r.dst = dst;
  r.bytes = opt.g2gBytes;
  r.linkToken = topology ? topology->linkFor(src, dst) : "UNKNOWN";
  r.linkDetail = describeTopologyToken(r.linkToken);

  void* srcBuf = nullptr;
  void* dstBuf = nullptr;
  void* srcBidirRecvBuf = nullptr;
  void* dstBidirSendBuf = nullptr;
  void* hostBuf = nullptr;
  cudaStream_t srcStream = nullptr;
  cudaStream_t dstStream = nullptr;

  std::string error;
  if (!allocateG2GBuffers(src, dst, r.bytes, srcBuf, dstBuf, error)) {
    r.error = error;
    return r;
  }

  if (!createPairStreams(src, dst, srcStream, dstStream, error)) {
    r.error = error;
    freePairBuffers(src, dst, srcBuf, dstBuf);
    destroyPairStreams(src, dst, srcStream, dstStream);
    return r;
  }

  r.peer = setupPeerAccessForPair(src, dst);

  cudaError_t he = cudaHostAlloc(&hostBuf, r.bytes, cudaHostAllocPortable);
  if (he != cudaSuccess) {
    cudaGetLastError();
    hostBuf = nullptr;
  }

  const double seconds = opt.effectiveG2GSeconds();
  const int batch = opt.g2gBatch;

  r.cudaCopy = runCudaMemcpyPeerBandwidth(src, dst, srcBuf, dstBuf, r.bytes, seconds, batch, dstStream);

  if (opt.g2gBidirectionalPair) {
    cudaError_t be = cudaSetDevice(src);
    if (be == cudaSuccess) be = cudaMalloc(&srcBidirRecvBuf, r.bytes);
    if (be == cudaSuccess) be = cudaMemset(srcBidirRecvBuf, 0x00, r.bytes);
    if (be == cudaSuccess) {
      be = cudaSetDevice(dst);
      if (be == cudaSuccess) be = cudaMalloc(&dstBidirSendBuf, r.bytes);
      if (be == cudaSuccess) be = cudaMemset(dstBidirSendBuf, 0xA5, r.bytes);
    }
    if (be == cudaSuccess) {
      std::vector<ConcurrentCopyTask> tasks;
      tasks.push_back({src, dst, srcBuf, dstBuf, dstStream});
      tasks.push_back({dst, src, dstBidirSendBuf, srcBidirRecvBuf, srcStream});
      r.cudaCopyBiDir = runConcurrentCudaMemcpyPeerBandwidth(
          tasks, r.bytes, seconds, batch, "bidirectional pair cudaMemcpyPeer");
    } else {
      r.cudaCopyBiDir.error = "bidirectional pair buffer allocation failed: " + cudaErrorString(be);
      cudaGetLastError();
    }
  } else {
    r.cudaCopyBiDir.error = "bidirectional pair copy not requested; use --g2g-bidir-pair";
  }

  if (r.peer.dstToSrcEnabled) {
    r.peerRead = runPeerRemoteReadBandwidth(src, dst, props, srcBuf, dstBuf, r.bytes, seconds, batch, dstStream);
  } else {
    r.peerRead.error = r.peer.dstCanAccessSrc ? "peer access dst->src could not be enabled"
                                              : "cudaDeviceCanAccessPeer(dst,src) is false";
  }

  if (r.peer.srcToDstEnabled) {
    r.peerWrite = runPeerRemoteWriteBandwidth(src, dst, props, srcBuf, dstBuf, r.bytes, seconds, batch, srcStream);
  } else {
    r.peerWrite.error = r.peer.srcCanAccessDst ? "peer access src->dst could not be enabled"
                                               : "cudaDeviceCanAccessPeer(src,dst) is false";
  }

  const bool anyCudaP2P = r.peer.srcCanAccessDst || r.peer.dstCanAccessSrc;
  if (hostBuf && (opt.g2gHostAlways || !anyCudaP2P || !r.cudaCopy.ran)) {
    r.hostStaged = runHostStagedBandwidth(src, dst, srcBuf, dstBuf, hostBuf,
                                          r.bytes, seconds, batch, srcStream, dstStream);
  } else if (hostBuf) {
    r.hostStaged.error = "host-staged path skipped because CUDA P2P is available; use --g2g-host-always to measure it";
  } else {
    r.hostStaged.error = "cudaHostAlloc pinned buffer failed: " + cudaErrorString(he);
  }

  if (nccl && nccl->hasSendRecv() && !ncclComms.empty()) {
    r.ncclSendRecv = runNcclSendRecvBandwidth(*nccl, ncclComms, src, dst, srcBuf, dstBuf,
                                              r.bytes, seconds, batch, srcStream, dstStream);
  } else if (nccl) {
    r.ncclSendRecv.error = "NCCL Send/Recv is not available in this NCCL runtime";
  } else {
    r.ncclSendRecv.error = "NCCL is not available";
  }

  auto consider = [&](const char* method, const std::string& path, const PairMethodResult& m) {
    if (m.ran && m.gbps > r.bestGBps) {
      r.bestGBps = m.gbps;
      r.bestMethod = method;
      r.bestPath = path;
    }
  };

  const std::string linkNote = r.linkToken.empty() ? "UNKNOWN" : r.linkToken;
  consider("CUDA_COPY", (r.peer.srcCanAccessDst || r.peer.dstCanAccessSrc)
                             ? "CUDA cudaMemcpyPeer, driver-selected P2P path over " + linkNote
                             : "CUDA cudaMemcpyPeer fallback path over " + linkNote,
           r.cudaCopy);
  consider("PEER_READ", "CUDA peer memory access over " + linkNote + ": dst kernel reads src memory", r.peerRead);
  consider("PEER_WRITE", "CUDA peer memory access over " + linkNote + ": src kernel writes dst memory", r.peerWrite);
  consider("HOST_STAGED", "Pinned host staged copy over PCIe/system memory", r.hostStaged);
  consider("NCCL_SENDRECV", "NCCL selected transport with local topology " + linkNote + ": P2P/SHM/NET as available", r.ncclSendRecv);

  if (!(r.bestGBps > 0.0)) {
    r.error = "no GPU-to-GPU method succeeded";
    appendError(r.error, r.cudaCopy.error);
    appendError(r.error, r.cudaCopyBiDir.error);
    appendError(r.error, r.peerRead.error);
    appendError(r.error, r.peerWrite.error);
    appendError(r.error, r.hostStaged.error);
    appendError(r.error, r.ncclSendRecv.error);
  }

  if (hostBuf) cudaFreeHost(hostBuf);
  if (srcBidirRecvBuf) {
    cudaSetDevice(src);
    cudaFree(srcBidirRecvBuf);
  }
  if (dstBidirSendBuf) {
    cudaSetDevice(dst);
    cudaFree(dstBidirSendBuf);
  }
  disablePeerAccessForPair(src, dst, r.peer);
  destroyPairStreams(src, dst, srcStream, dstStream);
  freePairBuffers(src, dst, srcBuf, dstBuf);
  return r;
}

static std::vector<GpuPairResult> runGpuToGpuBandwidthMatrix(
    int deviceCount, const std::vector<cudaDeviceProp>& props, NcclApi* nccl, const Options& opt,
    const GpuLinkTopology* topology) {
  std::vector<GpuPairResult> results;
  if (deviceCount < 2) return results;

  std::vector<int> devs(deviceCount);
  std::iota(devs.begin(), devs.end(), 0);

  std::vector<ncclComm_t_dyn> ncclComms(deviceCount, nullptr);
  bool ncclPairReady = false;
  if (nccl && nccl->hasSendRecv()) {
    int r = nccl->ncclCommInitAll(ncclComms.data(), deviceCount, devs.data());
    if (r == 0) {
      ncclPairReady = true;
      std::cout << "  NCCL Send/Recv pair test: enabled.\n";
    } else {
      std::cout << "  NCCL Send/Recv pair test: communicator init failed: "
                << nccl->err(r) << ". CUDA pair tests will still run.\n";
      std::fill(ncclComms.begin(), ncclComms.end(), nullptr);
    }
  } else if (nccl) {
    std::cout << "  NCCL Send/Recv pair test: unavailable; NCCL runtime has no ncclSend/ncclRecv symbols.\n";
  }

  results.reserve((size_t)deviceCount * (size_t)(deviceCount - 1));
  for (int src = 0; src < deviceCount; ++src) {
    for (int dst = 0; dst < deviceCount; ++dst) {
      if (src == dst) continue;
      const std::string linkToken = topology ? topology->linkFor(src, dst) : "UNKNOWN";
      std::cout << "  Testing GPU " << src << " -> GPU " << dst
                << " [" << linkToken << "] for "
                << std::fixed << std::setprecision(2) << opt.effectiveG2GSeconds()
                << " sec per method..." << std::flush;
      GpuPairResult r = runGpuToGpuPairBandwidth(src, dst, props,
                                                 ncclPairReady ? nccl : nullptr,
                                                 ncclComms, opt, topology);
      if (r.bestGBps > 0.0) {
        std::cout << " best " << r.bestMethod << " " << std::fixed << std::setprecision(2)
                  << r.bestGBps << " GB/s via " << r.linkToken << "\n";
      } else {
        std::cout << " failed: " << r.error << "\n";
      }
      results.push_back(std::move(r));
    }
  }

  if (ncclPairReady) {
    destroyNcclComms(*nccl, ncclComms);
  }
  return results;
}


struct PeerAccessMatrix {
  std::vector<std::vector<bool>> can;
  std::vector<std::vector<bool>> enabled;
  std::string error;
};

static PeerAccessMatrix setupPeerAccessAllPairs(int deviceCount) {
  PeerAccessMatrix m;
  m.can.assign((size_t)deviceCount, std::vector<bool>((size_t)deviceCount, false));
  m.enabled.assign((size_t)deviceCount, std::vector<bool>((size_t)deviceCount, false));
  for (int active = 0; active < deviceCount; ++active) {
    for (int peer = 0; peer < deviceCount; ++peer) {
      if (active == peer) continue;
      bool can = false;
      bool enabled = false;
      std::string err;
      if (!queryAndEnablePeerAccessDirection(active, peer, can, enabled, err)) {
        appendError(m.error, err);
      }
      m.can[(size_t)active][(size_t)peer] = can;
      m.enabled[(size_t)active][(size_t)peer] = enabled;
    }
  }
  return m;
}

static void disablePeerAccessAllPairs(const PeerAccessMatrix& m, int deviceCount) {
  for (int active = 0; active < deviceCount; ++active) {
    for (int peer = 0; peer < deviceCount; ++peer) {
      if (active == peer) continue;
      bool enabled = false;
      if (active < (int)m.enabled.size() && peer < (int)m.enabled[(size_t)active].size()) {
        enabled = m.enabled[(size_t)active][(size_t)peer];
      }
      disablePeerAccessDirection(active, peer, enabled);
    }
  }
}

struct G2GAggregateGpuResult {
  int gpu = -1;
  int peers = 0;
  size_t sliceBytes = 0;
  std::string linkSummary = "UNKNOWN";
  PairMethodResult oneToAll;
  PairMethodResult allToOne;
  PairMethodResult bidirectional;
  std::string error;
};

enum class AggregateCopyMode {
  OneToAll,
  AllToOne,
  Bidirectional
};

static int aggregatePeerSlot(int peer, int self) {
  return peer < self ? peer : peer - 1;
}

static void* aggregateSrcPtr(const std::vector<void*>& buffers, int src, int dst,
                             size_t sliceBytes, int deviceCount) {
  const int peers = deviceCount - 1;
  const size_t slot = (size_t)aggregatePeerSlot(dst, src);
  return static_cast<void*>(static_cast<char*>(buffers[(size_t)src]) + slot * sliceBytes);
}

static void* aggregateDstPtr(const std::vector<void*>& buffers, int src, int dst,
                             size_t sliceBytes, int deviceCount) {
  const int peers = deviceCount - 1;
  const size_t slot = (size_t)aggregatePeerSlot(src, dst);
  return static_cast<void*>(static_cast<char*>(buffers[(size_t)dst]) + ((size_t)peers + slot) * sliceBytes);
}

static void freeAggregateBuffers(std::vector<void*>& buffers) {
  for (int dev = 0; dev < (int)buffers.size(); ++dev) {
    if (!buffers[(size_t)dev]) continue;
    cudaSetDevice(dev);
    cudaFree(buffers[(size_t)dev]);
    buffers[(size_t)dev] = nullptr;
  }
}

static bool allocateAggregateBuffers(int deviceCount, size_t& sliceBytes,
                                     std::vector<void*>& buffers,
                                     std::string& error) {
  buffers.assign((size_t)deviceCount, nullptr);
  if (deviceCount < 2) {
    error = "aggregate GPU-to-GPU test needs at least 2 GPUs";
    return false;
  }

  sliceBytes = (sliceBytes / sizeof(uint4)) * sizeof(uint4);
  if (sliceBytes < MiB) sliceBytes = MiB;
  const int peers = deviceCount - 1;

  while (sliceBytes >= MiB) {
    error.clear();
    const size_t totalBytesPerGpu = sliceBytes * (size_t)peers * 2u;
    bool ok = true;
    for (int dev = 0; dev < deviceCount; ++dev) {
      cudaError_t e = cudaSetDevice(dev);
      if (e != cudaSuccess) {
        error = "cudaSetDevice failed on GPU " + std::to_string(dev) +
                " while allocating aggregate buffers: " + cudaErrorString(e);
        ok = false;
        break;
      }
      e = cudaMalloc(&buffers[(size_t)dev], totalBytesPerGpu);
      if (e != cudaSuccess) {
        cudaGetLastError();
        error = "cudaMalloc aggregate buffer failed on GPU " + std::to_string(dev) +
                " for " + bytesToGiB(totalBytesPerGpu) + ": " + cudaErrorString(e);
        ok = false;
        break;
      }
      e = cudaMemset(buffers[(size_t)dev], 0x3C, totalBytesPerGpu);
      if (e == cudaSuccess) e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        error = "cudaMemset/sync aggregate buffer failed on GPU " + std::to_string(dev) +
                ": " + cudaErrorString(e);
        ok = false;
        break;
      }
    }
    if (ok) return true;
    freeAggregateBuffers(buffers);
    sliceBytes /= 2;
    sliceBytes = (sliceBytes / sizeof(uint4)) * sizeof(uint4);
  }

  if (error.empty()) error = "aggregate GPU-to-GPU buffer allocation failed; slice dropped below 1 MiB";
  return false;
}

static PairMethodResult runAggregateCopyModeForGpu(
    int focusGpu, int deviceCount, const std::vector<void*>& buffers,
    size_t sliceBytes, double seconds, int batch, AggregateCopyMode mode) {
  PairMethodResult m;
  std::vector<ConcurrentCopyTask> tasks;
  tasks.reserve((size_t)(deviceCount - 1) * 2u);

  for (int peer = 0; peer < deviceCount; ++peer) {
    if (peer == focusGpu) continue;
    if (mode == AggregateCopyMode::OneToAll || mode == AggregateCopyMode::Bidirectional) {
      tasks.push_back({focusGpu, peer,
                       aggregateSrcPtr(buffers, focusGpu, peer, sliceBytes, deviceCount),
                       aggregateDstPtr(buffers, focusGpu, peer, sliceBytes, deviceCount),
                       nullptr});
    }
    if (mode == AggregateCopyMode::AllToOne || mode == AggregateCopyMode::Bidirectional) {
      tasks.push_back({peer, focusGpu,
                       aggregateSrcPtr(buffers, peer, focusGpu, sliceBytes, deviceCount),
                       aggregateDstPtr(buffers, peer, focusGpu, sliceBytes, deviceCount),
                       nullptr});
    }
  }

  std::string err;
  if (!createConcurrentCopyStreams(tasks, err)) {
    m.error = err;
    destroyConcurrentCopyStreams(tasks);
    return m;
  }

  const char* label = mode == AggregateCopyMode::OneToAll
                          ? "aggregate one-to-all cudaMemcpyPeer"
                          : (mode == AggregateCopyMode::AllToOne
                                 ? "aggregate all-to-one cudaMemcpyPeer"
                                 : "aggregate bidirectional cudaMemcpyPeer");
  m = runConcurrentCudaMemcpyPeerBandwidth(tasks, sliceBytes, seconds, batch, label);
  destroyConcurrentCopyStreams(tasks);
  return m;
}

static std::vector<G2GAggregateGpuResult> runG2GAggregateBandwidth(int deviceCount,
                                                                   const Options& opt,
                                                                   const GpuLinkTopology* topology) {
  std::vector<G2GAggregateGpuResult> results;
  if (deviceCount < 2) return results;
  results.reserve((size_t)deviceCount);

  PeerAccessMatrix peerMatrix = setupPeerAccessAllPairs(deviceCount);
  if (!peerMatrix.error.empty()) {
    std::cout << "  Peer access warning: " << peerMatrix.error << "\n";
  }

  size_t sliceBytes = opt.g2gBytes;
  std::vector<void*> buffers;
  std::string error;
  if (!allocateAggregateBuffers(deviceCount, sliceBytes, buffers, error)) {
    std::cout << "  Aggregate GPU-to-GPU test failed during allocation: " << error << "\n";
    disablePeerAccessAllPairs(peerMatrix, deviceCount);
    return results;
  }

  const double seconds = opt.effectiveG2GAggregateSeconds();
  const int batch = opt.g2gBatch;
  for (int gpu = 0; gpu < deviceCount; ++gpu) {
    G2GAggregateGpuResult r;
    r.gpu = gpu;
    r.peers = deviceCount - 1;
    r.sliceBytes = sliceBytes;
    r.linkSummary = topology ? summarizeLinksForGpu(*topology, gpu, deviceCount) : "UNKNOWN";
    std::cout << "  Aggregate GPU " << gpu << " [" << r.linkSummary << "] for "
              << std::fixed << std::setprecision(2)
              << seconds << " sec per aggregate mode..." << std::flush;
    r.oneToAll = runAggregateCopyModeForGpu(gpu, deviceCount, buffers, sliceBytes,
                                            seconds, batch, AggregateCopyMode::OneToAll);
    r.allToOne = runAggregateCopyModeForGpu(gpu, deviceCount, buffers, sliceBytes,
                                            seconds, batch, AggregateCopyMode::AllToOne);
    r.bidirectional = runAggregateCopyModeForGpu(gpu, deviceCount, buffers, sliceBytes,
                                                 seconds, batch, AggregateCopyMode::Bidirectional);
    if (!r.oneToAll.ran) appendError(r.error, "one-to-all: " + r.oneToAll.error);
    if (!r.allToOne.ran) appendError(r.error, "all-to-one: " + r.allToOne.error);
    if (!r.bidirectional.ran) appendError(r.error, "bidirectional: " + r.bidirectional.error);

    if (r.bidirectional.ran) {
      std::cout << " bidir " << std::fixed << std::setprecision(2)
                << r.bidirectional.gbps << " GB/s";
    }
    if (!r.error.empty()) std::cout << " warning: " << r.error;
    std::cout << "\n";
    results.push_back(std::move(r));
  }

  freeAggregateBuffers(buffers);
  disablePeerAccessAllPairs(peerMatrix, deviceCount);
  return results;
}

static void printG2GAggregateSummary(const std::vector<G2GAggregateGpuResult>& results) {
  if (results.empty()) return;
  std::cout << "\n================ GPU-to-GPU Aggregate Summary ================\n";
  std::cout << "Aggregate tests launch multiple cudaMemcpyPeer transfers at the same time.\n";
  std::cout << "OneToAll and AllToOne are one-way aggregate payload bandwidth. BiDirAgg is send+receive payload bandwidth and is the closest number to compare with a bidirectional per-GPU NVLink spec.\n";
  std::cout << std::left << std::setw(5) << "GPU"
            << std::setw(8) << "Peers"
            << std::setw(18) << "Links"
            << std::setw(12) << "Slice"
            << std::right << std::setw(15) << "OneToAll"
            << std::setw(15) << "AllToOne"
            << std::setw(15) << "BiDirAgg"
            << "  Notes\n";
  std::cout << std::string(106, '-') << "\n";
  for (const auto& r : results) {
    std::cout << std::left << std::setw(5) << r.gpu
              << std::setw(8) << r.peers
              << std::setw(18) << fitColumn(r.linkSummary, 17)
              << std::setw(12) << bytesToGiB(r.sliceBytes)
              << std::right << std::setw(15) << speedToString(r.oneToAll.gbps)
              << std::setw(15) << speedToString(r.allToOne.gbps)
              << std::setw(15) << speedToString(r.bidirectional.gbps)
              << "  " << (r.error.empty() ? "" : r.error) << "\n";
  }
  std::cout << std::string(106, '-') << "\n";
  std::cout << "For H100 SXM/HGX, 900 GB/s is a bidirectional aggregate per-GPU number, not a single directed src->dst copy number.\n";
}

static void printGpuToGpuSummary(const std::vector<GpuPairResult>& results, int deviceCount) {
  if (results.empty()) return;

  std::cout << "\n================ GPU-to-GPU Pair Summary ================\n";
  std::cout << "Directional pair tests. Read means dst GPU kernel reads src GPU memory; "
            << "Write means src GPU kernel writes dst GPU memory.\n";
  std::cout << std::left << std::setw(5) << "Src"
            << std::setw(5) << "Dst"
            << std::setw(12) << "Size"
            << std::setw(11) << "P2P"
            << std::setw(12) << "Link"
            << std::right << std::setw(12) << "Copy"
            << std::setw(12) << "BiDirCopy"
            << std::setw(12) << "Read"
            << std::setw(12) << "Write"
            << std::setw(12) << "NCCL"
            << std::setw(12) << "Host"
            << std::setw(12) << "Best"
            << "  BestMethod\n";
  std::cout << std::string(146, '-') << "\n";

  const GpuPairResult* fastest = nullptr;
  for (const auto& r : results) {
    if (!fastest || r.bestGBps > fastest->bestGBps) fastest = &r;
    std::string p2p = std::string(r.peer.srcCanAccessDst ? "Y" : "N") + "/" +
                      (r.peer.dstCanAccessSrc ? "Y" : "N");
    std::cout << std::left << std::setw(5) << r.src
              << std::setw(5) << r.dst
              << std::setw(12) << bytesToGiB(r.bytes)
              << std::setw(11) << p2p
              << std::setw(12) << fitColumn(r.linkToken, 11)
              << std::right << std::setw(12) << speedToString(r.cudaCopy.gbps)
              << std::setw(12) << speedToString(r.cudaCopyBiDir.gbps)
              << std::setw(12) << speedToString(r.peerRead.gbps)
              << std::setw(12) << speedToString(r.peerWrite.gbps)
              << std::setw(12) << speedToString(r.ncclSendRecv.gbps)
              << std::setw(12) << speedToString(r.hostStaged.gbps)
              << std::setw(12) << speedToString(r.bestGBps)
              << "  " << (r.bestMethod.empty() ? "n/a" : r.bestMethod) << "\n";
  }
  std::cout << std::string(146, '-') << "\n";
  std::cout << "P2P column is src->dst / dst->src cudaDeviceCanAccessPeer. Link comes from nvidia-smi topo -m when available.\n";
  std::cout << "Copy is one directed cudaMemcpyPeer payload GB/s. BiDirCopy is simultaneous src->dst plus dst->src total payload GB/s when --g2g-bidir-pair is used. Host is logical payload GB/s through pinned host memory.\n";
  std::cout << "BestMethod chooses the fastest measured method for that directed pair.\n";

  std::vector<std::vector<double>> best((size_t)deviceCount, std::vector<double>((size_t)deviceCount, 0.0));
  for (const auto& r : results) {
    if (r.src >= 0 && r.src < deviceCount && r.dst >= 0 && r.dst < deviceCount) {
      best[(size_t)r.src][(size_t)r.dst] = r.bestGBps;
    }
  }

  std::cout << "\nBest GPU-to-GPU payload bandwidth matrix, GB/s, rows are source GPUs, columns are destination GPUs:\n";
  std::cout << std::setw(8) << "src\\dst";
  for (int dst = 0; dst < deviceCount; ++dst) {
    std::ostringstream label;
    label << "GPU" << dst;
    std::cout << std::setw(10) << label.str();
  }
  std::cout << "\n";
  for (int src = 0; src < deviceCount; ++src) {
    std::ostringstream label;
    label << "GPU" << src;
    std::cout << std::setw(8) << label.str();
    for (int dst = 0; dst < deviceCount; ++dst) {
      if (src == dst) {
        std::cout << std::setw(10) << "--";
      } else {
        std::cout << std::setw(10) << speedToString(best[(size_t)src][(size_t)dst]);
      }
    }
    std::cout << "\n";
  }

  if (fastest && fastest->bestGBps > 0.0) {
    std::cout << "Fastest directed GPU-to-GPU pair: GPU " << fastest->src << " -> GPU "
              << fastest->dst << ", " << std::fixed << std::setprecision(2)
              << fastest->bestGBps << " GB/s by " << fastest->bestMethod
              << ", link " << fastest->linkToken << " (" << fastest->linkDetail << ")"
              << " (" << fastest->bestPath << ").\n";
  }
}

int main(int argc, char** argv) {
  Options opt;
  if (!parseArgs(argc, argv, opt)) {
    printUsage(argv[0]);
    return 2;
  }

  int deviceCount = 0;
  cudaError_t e = cudaGetDeviceCount(&deviceCount);
  if (e != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaErrorString(e) << "\n";
    return 1;
  }
  if (deviceCount <= 0) {
    std::cerr << "No CUDA GPUs found.\n";
    return 1;
  }

  std::cout << "CUDA GPUs: " << deviceCount << "\n";
  std::cout << "Single-GPU memory stress duration: " << opt.singleSeconds << " sec\n";
  std::cout << "All-GPU memory/NCCL stress duration: " << opt.effectiveAllSeconds() << " sec";
  if (opt.allSeconds <= 0.0) std::cout << " (same as single-GPU)";
  std::cout << "\n";
  std::cout << "Memory reserve per GPU: " << bytesToGiB(opt.reserveBytes) << "\n";
  if (opt.g2gTest && deviceCount >= 2) {
    std::cout << "GPU-to-GPU pair test: " << (opt.g2gPairTest ? "enabled" : "disabled")
              << ", per method/directed pair=" << opt.effectiveG2GSeconds()
              << " sec, buffer=" << bytesToGiB(opt.g2gBytes)
              << ", batch=" << opt.g2gBatch
              << ", host-staged=" << (opt.g2gHostAlways ? "always" : "fallback-only")
              << ", bidir-pair=" << (opt.g2gBidirectionalPair ? "enabled" : "disabled") << "\n";
    std::cout << "GPU-to-GPU aggregate test: "
              << (opt.g2gAggregateTest ? "enabled" : "disabled")
              << ", per focus GPU/mode=" << opt.effectiveG2GAggregateSeconds()
              << " sec\n";
  } else if (deviceCount < 2) {
    std::cout << "GPU-to-GPU pair test: skipped; at least 2 GPUs are required.\n";
  } else {
    std::cout << "GPU-to-GPU pair test: disabled\n";
  }
  if (opt.tempMonitor) {
    std::cout << "Temperature monitor: enabled, threshold=" << std::fixed << std::setprecision(1)
              << opt.tempThresholdC << " C, interval=" << opt.tempIntervalMs << " ms\n";
  } else {
    std::cout << "Temperature monitor: disabled\n";
  }

  NcclApi nccl;
  bool ncclAvailable = nccl.load();
  if (ncclAvailable) {
    std::cout << "NCCL: available, version code " << nccl.version << "\n";
  } else {
    std::cout << "NCCL: not available (" << nccl.loadError << "). CUDA memory stress will still run.\n";
  }

  std::vector<cudaDeviceProp> props(deviceCount);
  std::cout << "\nDetected GPU memory:\n";
  for (int i = 0; i < deviceCount; ++i) {
    cudaSetDevice(i);
    cudaGetDeviceProperties(&props[i], i);
    std::cout << "  GPU " << i << "  " << props[i].name
              << "  total=" << bytesToGiB(props[i].totalGlobalMem) << "\n";
  }

  GpuLinkTopology g2gTopology;
  if (deviceCount >= 2) {
    g2gTopology = buildGpuLinkTopology(deviceCount);
    printGpuLinkTopologyMatrix(g2gTopology, deviceCount);
  }

  NvmlApi nvmlTemp;
  std::vector<nvmlDevice_t_dyn> nvmlTempHandles;
  std::vector<TempStatus> tempStatus(deviceCount);
  std::atomic<bool> stopTempMonitor{false};
  std::thread tempMonitorThread;
  bool tempMonitorRunning = false;

  if (opt.tempMonitor) {
    std::string tempError;
    if (nvmlTemp.load() && initNvmlTemperatureHandles(nvmlTemp, deviceCount, nvmlTempHandles,
                                                       tempStatus, tempError)) {
      std::cout << "Temperature monitor: NVML ready; warnings will be printed when a GPU reaches "
                << std::fixed << std::setprecision(1) << opt.tempThresholdC << " C or higher.\n";
      tempMonitorThread = std::thread(temperatureMonitorLoop, &nvmlTemp, &nvmlTempHandles, &props,
                                      opt.tempThresholdC, opt.tempIntervalMs,
                                      &stopTempMonitor, &tempStatus);
      tempMonitorRunning = true;
    } else {
      if (tempError.empty()) tempError = nvmlTemp.loadError;
      std::cout << "Temperature monitor: unavailable (" << tempError << "). Tests will continue.\n";
    }
  }

  std::cout << "\nSequential memory stress, one GPU at a time...\n";
  std::vector<MemResult> seq(deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    std::cout << "  Testing GPU " << i << " for " << opt.singleSeconds << " sec..." << std::flush;
    seq[i] = runMemoryStressOneGpu(i, opt.singleSeconds, opt.reserveBytes);
    if (seq[i].error.empty()) {
      std::cout << " done, " << std::fixed << std::setprecision(2) << seq[i].gbps << " GB/s\n";
    } else {
      std::cout << " failed: " << seq[i].error << "\n";
    }
  }

  std::cout << "\nConcurrent memory stress, all GPUs together for "
            << opt.effectiveAllSeconds() << " sec...\n";
  std::vector<MemResult> all(deviceCount);
  std::vector<std::thread> threads;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};

  for (int i = 0; i < deviceCount; ++i) {
    threads.emplace_back([&, i] {
      all[i] = runMemoryStressOneGpu(i, opt.effectiveAllSeconds(), opt.reserveBytes, &ready, &go);
    });
  }
  while (ready.load(std::memory_order_acquire) < deviceCount) std::this_thread::yield();
  go.store(true, std::memory_order_release);
  for (auto& t : threads) t.join();

  double totalAllGpuGBps = 0.0;
  for (const auto& r : all) {
    if (r.error.empty()) totalAllGpuGBps += r.gbps;
  }
  std::cout << "  All-GPU total memory stress bandwidth: " << std::fixed << std::setprecision(2)
            << totalAllGpuGBps << " GB/s\n";

  NcclResult nr;
  if (ncclAvailable) {
    std::cout << "\nNCCL AllReduce link test, all GPUs together for "
              << opt.effectiveAllSeconds() << " sec...\n";
    nr = runNcclAllReduce(nccl, deviceCount, opt);
    if (nr.ran) {
      std::cout << "  NCCL buffer per GPU: " << bytesToGiB(nr.bytesPerRank) << "\n"
                << "  NCCL iterations: " << nr.iterations << ", elapsed=" << std::fixed
                << std::setprecision(3) << nr.elapsedSec << " sec\n"
                << "  NCCL AllReduce algBW: " << std::fixed << std::setprecision(2)
                << nr.algGBps << " GB/s\n"
                << "  NCCL AllReduce busBW/link estimate: " << std::fixed << std::setprecision(2)
                << nr.busGBps << " GB/s\n";
    } else {
      std::cout << "  NCCL test skipped/failed: " << nr.error << "\n";
    }
  }

  std::vector<GpuPairResult> g2gResults;
  std::vector<G2GAggregateGpuResult> g2gAggregateResults;
  if (opt.g2gTest && deviceCount >= 2) {
    if (opt.g2gPairTest) {
      std::cout << "\nGPU-to-GPU pair bandwidth test, all directed GPU pairs, "
                << std::fixed << std::setprecision(2) << opt.effectiveG2GSeconds()
                << " sec per method, buffer=" << bytesToGiB(opt.g2gBytes) << "...\n";
      std::cout << "  Methods: cudaMemcpyPeer, peer remote read, peer remote write, pinned host-staged copy";
      if (opt.g2gBidirectionalPair) std::cout << ", bidirectional cudaMemcpyPeer";
      if (ncclAvailable) std::cout << ", NCCL Send/Recv when supported";
      std::cout << ".\n";
      g2gResults = runGpuToGpuBandwidthMatrix(deviceCount, props, ncclAvailable ? &nccl : nullptr, opt, &g2gTopology);
    }

    if (opt.g2gAggregateTest) {
      std::cout << "\nGPU-to-GPU aggregate bandwidth test, one focus GPU at a time, "
                << std::fixed << std::setprecision(2) << opt.effectiveG2GAggregateSeconds()
                << " sec per aggregate mode, slice=" << bytesToGiB(opt.g2gBytes) << "...\n";
      std::cout << "  Modes: one-to-all, all-to-one, bidirectional aggregate cudaMemcpyPeer.\n";
      g2gAggregateResults = runG2GAggregateBandwidth(deviceCount, opt, &g2gTopology);
    }
  }

  if (tempMonitorRunning) {
    stopTempMonitor.store(true, std::memory_order_release);
    tempMonitorThread.join();
    tempMonitorRunning = false;
  }

  std::cout << "\n================ Summary ================\n";
  std::cout << std::left << std::setw(5) << "GPU"
            << std::setw(32) << "Name"
            << std::right << std::setw(12) << "VRAM"
            << std::setw(14) << "Alloc"
            << std::setw(16) << "SeqMemGB/s"
            << std::setw(16) << "AllMemGB/s"
            << std::setw(11) << "MaxTempC"
            << std::setw(12) << "TempAlert" << "\n";
  std::cout << std::string(118, '-') << "\n";

  for (int i = 0; i < deviceCount; ++i) {
    std::string name = props[i].name;
    if (name.size() > 31) name = name.substr(0, 31);
    std::cout << std::left << std::setw(5) << i
              << std::setw(32) << name
              << std::right << std::setw(12) << bytesToGiB(props[i].totalGlobalMem)
              << std::setw(14) << (seq[i].allocatedBytes ? bytesToGiB(seq[i].allocatedBytes) : "n/a");

    if (seq[i].error.empty()) std::cout << std::setw(16) << std::fixed << std::setprecision(2) << seq[i].gbps;
    else std::cout << std::setw(16) << "ERR";

    if (all[i].error.empty()) std::cout << std::setw(16) << std::fixed << std::setprecision(2) << all[i].gbps;
    else std::cout << std::setw(16) << "ERR";

    if (i < (int)tempStatus.size() && tempStatus[i].hasReading) {
      std::cout << std::setw(11) << (std::to_string(tempStatus[i].maxC) + " C")
                << std::setw(12) << (tempStatus[i].everOverThreshold ? "YES" : "NO");
    } else if (i < (int)tempStatus.size() && !tempStatus[i].error.empty()) {
      std::cout << std::setw(11) << "ERR" << std::setw(12) << "n/a";
    } else {
      std::cout << std::setw(11) << "n/a" << std::setw(12) << "n/a";
    }
    std::cout << "\n";
  }

  std::cout << std::string(118, '-') << "\n";
  std::cout << "Single-GPU test duration: " << opt.singleSeconds << " sec\n";
  std::cout << "All-GPU test duration: " << opt.effectiveAllSeconds() << " sec";
  if (opt.allSeconds <= 0.0) std::cout << " (same as single-GPU)";
  std::cout << "\n";
  std::cout << "All-GPU memory total: " << std::fixed << std::setprecision(2)
            << totalAllGpuGBps << " GB/s\n";
  if (opt.g2gTest && deviceCount >= 2) {
    if (opt.g2gPairTest) {
      std::cout << "GPU-to-GPU pair test duration per method/directed pair: "
                << std::fixed << std::setprecision(2) << opt.effectiveG2GSeconds() << " sec\n";
      std::cout << "GPU-to-GPU pair buffer: " << bytesToGiB(opt.g2gBytes) << "\n";
    }
    if (opt.g2gAggregateTest) {
      std::cout << "GPU-to-GPU aggregate test duration per focus GPU/mode: "
                << std::fixed << std::setprecision(2) << opt.effectiveG2GAggregateSeconds() << " sec\n";
    }
  }

  if (opt.tempMonitor) {
    bool anyTempReading = false;
    bool anyTempAlert = false;
    for (const auto& st : tempStatus) {
      anyTempReading = anyTempReading || st.hasReading;
      anyTempAlert = anyTempAlert || st.everOverThreshold;
    }
    if (anyTempReading) {
      std::cout << "Temperature threshold: " << std::fixed << std::setprecision(1)
                << opt.tempThresholdC << " C; alert triggered: "
                << (anyTempAlert ? "YES" : "NO") << "\n";
    } else {
      std::cout << "Temperature monitor: no readings collected.\n";
    }
  }

  printGpuToGpuSummary(g2gResults, deviceCount);
  printG2GAggregateSummary(g2gAggregateResults);

  if (ncclAvailable) {
    if (nr.ran) {
      std::cout << "NCCL version code: " << nr.version << "\n"
                << "NCCL AllReduce buffer/rank: " << bytesToGiB(nr.bytesPerRank) << "\n"
                << "NCCL AllReduce algBW: " << std::fixed << std::setprecision(2)
                << nr.algGBps << " GB/s\n"
                << "NCCL AllReduce busBW/link estimate: " << std::fixed << std::setprecision(2)
                << nr.busGBps << " GB/s\n";
    } else {
      std::cout << "NCCL available but test did not run: " << nr.error << "\n";
    }
  } else {
    std::cout << "NCCL: not available; link speed not measured.\n";
  }

  bool ok = true;
  for (const auto& r : seq) if (!r.error.empty()) ok = false;
  for (const auto& r : all) if (!r.error.empty()) ok = false;
  if (ncclAvailable && !nr.ran && deviceCount >= 2) ok = false;
  if (opt.g2gTest && deviceCount >= 2) {
    if (opt.g2gPairTest) {
      if (g2gResults.empty()) ok = false;
      for (const auto& r : g2gResults) {
        if (!(r.bestGBps > 0.0)) ok = false;
      }
    }
    if (opt.g2gAggregateTest) {
      if (g2gAggregateResults.empty()) ok = false;
      for (const auto& r : g2gAggregateResults) {
        if (!r.oneToAll.ran || !r.allToOne.ran || !r.bidirectional.ran) ok = false;
      }
    }
  }
  return ok ? 0 : 1;
}
