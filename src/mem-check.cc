#include "addr2sym.hpp"
#include "alloc-action.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <new>

#include "addr2sym.hpp"
#include "alloc-action.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <new>

#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <memory_resource>

#ifdef __ENABLE_CUDA__
#include <cuda_runtime.h>
#include <dlfcn.h>

#endif

__attribute__((constructor)) void preload_init() {
  fprintf(stdout, "mem-check library loaded\n");
}

namespace detail {

struct GlobalData {
  size_t const buff_size{64 * 1024 * 1024};
  void *buff{nullptr};
  std::pmr::monotonic_buffer_resource mono;
  std::pmr::unsynchronized_pool_resource pool;
  std::pmr::map<void *, AllocAction> allocated;
  std::mutex lock{};
  bool enable{false};

  GlobalData()
      : buff(mmap(nullptr, buff_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)),
        mono(buff, buff_size), pool(&mono), allocated(&pool), enable(true) {
    if (buff == MAP_FAILED) {
      perror("mmap");
      std::abort();
    }
  }

  ~GlobalData() {
    enable = false;
    for (auto &&[ptr, info] : allocated) {
      printf("检测到内存泄漏 ptr = %p, size = %zd, caller = %s, action = %s\n",
             ptr, info.size, addr2sym(info.caller).c_str(),
             kAllocOpNames[static_cast<size_t>(info.op)]);
    }
  }

  void on(AllocOp op, void *ptr, size_t size, size_t align, void *caller,
          int64_t timestamp) {
    if (!ptr) {
      return;
    }
    if (kAllocOpIsAllocation[static_cast<size_t>(op)]) {

      if (allocated.find(ptr) != allocated.end()) {
        printf("检测到内存多次分配同一个地址 ptr = %p, size = %zd, "
               "first caller = %s, this caller = %s\n",
               ptr, size, addr2sym(allocated[ptr].caller).c_str(),
               addr2sym(caller).c_str());
      } else {
        std::pair<void *, AllocAction> data = {
            ptr, AllocAction{op, 0, ptr, size, align, caller, timestamp}};
        allocated.insert(data);
      }
    } else {
      auto it = allocated.find(ptr);
      if (it == allocated.end()) {
        // TODO : this has problem, because you can't ensure the global data is
        // initialized firstly.
        /*
        printf("检测到尝试释放不存在的内存 ptr = %p, size = %zd, "
               "caller = %s, action = %s\n",
               ptr, size, addr2sym(caller).c_str(),
               kAllocOpNames[static_cast<size_t>(op)]);
        */
        return;
      }

      const AllocAction &action = it->second;
      bool wrong_free_func =
          kAllocOpFreeFunction[static_cast<size_t>(action.op)] != op;

      if (wrong_free_func) {
        printf("检测到内存释放错误 ptr = %p, size = %zd, caller = %s, alloc op "
               "= %s, free alloc = %s\n",
               ptr, action.size, addr2sym(caller).c_str(),
               kAllocOpNames[static_cast<size_t>(action.op)],
               kAllocOpNames[static_cast<size_t>(op)]);
      }
      allocated.erase(it);
    }
  }
};

} // namespace detail

static detail::GlobalData global_data;
static detail::GlobalData *global = nullptr;
static int global_init_helper = (global = &global_data, 0);

struct EnableGuard {

  bool was_enabled = false;

  EnableGuard() {
    if (!global) {
      was_enabled = false;
    } else {
      was_enabled = global->enable;
      global->enable = false;
    }
    if (was_enabled) {
      global->lock.lock();
    }
  }

  void on(AllocOp op, void *ptr, size_t size, size_t align, void *caller) {
    if (ptr && was_enabled) {
      auto now = std::chrono::high_resolution_clock::now();
      int64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              now.time_since_epoch())
                              .count();
      EnableGuard ena;
      global->on(op, ptr, size, align, caller, timestamp);
    };
  }

  ~EnableGuard() {
    if (global && was_enabled) {
      global->enable = was_enabled;
      global->lock.unlock();
    }
  }
};

void operator delete(void *ptr) noexcept(false) {
  EnableGuard ena;
  ena.on(AllocOp::Delete, ptr, kNone, kNone, __builtin_return_address(0));
  free(ptr);
}

void operator delete[](void *ptr) noexcept(false) {

  EnableGuard ena;
  ena.on(AllocOp::DeleteArray, ptr, kNone, kNone, __builtin_return_address(0));
  free(ptr);
}

void *operator new(size_t size) noexcept(false) {
  EnableGuard ena;
  void *ptr = malloc(size);
  ena.on(AllocOp::New, ptr, size, kNone, __builtin_return_address(0));

  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new[](size_t size) noexcept(false) {
  EnableGuard ena;
  void *ptr = malloc(size);
  ena.on(AllocOp::NewArray, ptr, size, kNone, __builtin_return_address(0));

  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new(size_t size, std::nothrow_t const &) noexcept {
  EnableGuard ena;
  void *ptr = malloc(size);
  ena.on(AllocOp::New, ptr, size, kNone, __builtin_return_address(0));
  return ptr;
}

void *operator new[](size_t size, std::nothrow_t const &) noexcept {

  EnableGuard ena;
  void *ptr = malloc(size);
  ena.on(AllocOp::NewArray, ptr, size, kNone, __builtin_return_address(0));

  return ptr;
}

void operator delete(void *ptr, size_t size) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::Delete, ptr, size, kNone, __builtin_return_address(0));

  free(ptr);
}

void operator delete[](void *ptr, size_t size) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::DeleteArray, ptr, size, kNone, __builtin_return_address(0));

  free(ptr);
}

void operator delete(void *ptr, std::align_val_t align) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::Delete, ptr, kNone, static_cast<size_t>(align),
         __builtin_return_address(0));

  free(ptr);
}

void operator delete[](void *ptr, std::align_val_t align) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::DeleteArray, ptr, kNone, static_cast<size_t>(align),
         __builtin_return_address(0));

  free(ptr);
}

void operator delete(void *ptr, std::size_t size,
                     std::align_val_t align) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::Delete, ptr, size, static_cast<size_t>(align),
         __builtin_return_address(0));

  free(ptr);
}

void operator delete[](void *ptr, std::size_t size,
                       std::align_val_t align) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::DeleteArray, ptr, size, static_cast<size_t>(align),
         __builtin_return_address(0));

  free(ptr);
}

void operator delete(void *ptr, std::align_val_t align,
                     std::nothrow_t const &) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::Delete, ptr, kNone, static_cast<size_t>(align),
         __builtin_return_address(0));

  free(ptr);
}

void operator delete[](void *ptr, std::align_val_t align,
                       std::nothrow_t const &) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::DeleteArray, ptr, kNone, static_cast<size_t>(align),
         __builtin_return_address(0));

  free(ptr);
}

void *operator new(std::size_t size, std::align_val_t align) noexcept(false) {
  EnableGuard ena;
  void *ptr = aligned_alloc(static_cast<size_t>(align), size);
  ena.on(AllocOp::New, ptr, size, static_cast<size_t>(align),
         __builtin_return_address(0));

  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new[](size_t size, std::align_val_t align) noexcept(false) {

  EnableGuard ena;
  void *ptr = aligned_alloc(static_cast<size_t>(align), size);
  ena.on(AllocOp::NewArray, ptr, size, static_cast<size_t>(align),
         __builtin_return_address(0));

  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new(std::size_t size, std::align_val_t align,
                   std::nothrow_t const &) noexcept {

  EnableGuard ena;
  void *ptr = aligned_alloc(static_cast<size_t>(align), size);
  ena.on(AllocOp::New, ptr, size, static_cast<size_t>(align),
         __builtin_return_address(0));

  return ptr;
}

void *operator new[](size_t size, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
  EnableGuard ena;
  void *ptr = aligned_alloc(static_cast<size_t>(align), size);
  ena.on(AllocOp::NewArray, ptr, size, static_cast<size_t>(align),
         __builtin_return_address(0));

  return ptr;
}

extern "C" void *__libc_malloc(size_t size) noexcept;
extern "C" void *__libc_free(void *ptr) noexcept;

extern "C" void *__libc_calloc(size_t nmemb, size_t size) noexcept;
extern "C" void *__libc_realloc(void *ptr, size_t size) noexcept;
extern "C" void *__libc_reallocarray(void *ptr, size_t nmemb,
                                     size_t size) noexcept;

extern "C" void *__libc_valloc(size_t size) noexcept;
extern "C" void *__libc_memalign(size_t alignment, size_t size) noexcept;

#define REAL_LIBC(name) __libc_##name

#define RETURN_ADDRESS __builtin_return_address(0)

// c
extern "C" void *malloc(size_t size) noexcept {
  EnableGuard ena;
  void *ptr = REAL_LIBC(malloc)(size);
  ena.on(AllocOp::Malloc, ptr, size, kNone, RETURN_ADDRESS);

  return ptr;
}

extern "C" void free(void *ptr) noexcept {

  EnableGuard ena;
  ena.on(AllocOp::Free, ptr, 0, kNone, RETURN_ADDRESS);

  REAL_LIBC(free)(ptr);
}

extern "C" void *calloc(size_t nmemb, size_t size) noexcept {
  EnableGuard ena;
  void *ptr = REAL_LIBC(calloc)(nmemb, size);
  ena.on(AllocOp::Malloc, ptr, nmemb * size, kNone, RETURN_ADDRESS);

  return ptr;
}

extern "C" void *realloc(void *ptr, size_t size) noexcept {
  EnableGuard ena;
  void *new_ptr = REAL_LIBC(realloc)(ptr, size);
  ena.on(AllocOp::Malloc, new_ptr, size, kNone, RETURN_ADDRESS);

  return new_ptr;
}

extern "C" void *reallocarray(void *ptr, size_t nmemb, size_t size) noexcept {
  EnableGuard ena;
  void *new_ptr = REAL_LIBC(reallocarray)(ptr, nmemb, size);
  ena.on(AllocOp::Malloc, new_ptr, nmemb * size, kNone, RETURN_ADDRESS);

  return new_ptr;
}

extern "C" void *valloc(size_t size) noexcept {
  EnableGuard ena;
  void *ptr = REAL_LIBC(valloc)(size);
  size_t pagesize = sysconf(_SC_PAGESIZE);
  ena.on(AllocOp::Malloc, ptr, size, pagesize, RETURN_ADDRESS);

  return ptr;
}

extern "C" void *memalign(size_t alignment, size_t size) noexcept {
  EnableGuard ena;
  void *ptr = REAL_LIBC(memalign)(alignment, size);
  ena.on(AllocOp::Malloc, ptr, size, alignment, RETURN_ADDRESS);

  return ptr;
}

extern "C" int posix_memalign(void **memptr, size_t alignment, size_t size) {
  EnableGuard ena;
  void *ptr = REAL_LIBC(memalign)(alignment, size);
  ena.on(AllocOp::Malloc, *memptr, size, alignment, RETURN_ADDRESS);

  int ret = 0;
  if (!ptr) {
    ret = errno;
  } else {
    *memptr = ptr;
  }
  return ret;
}

#ifdef __ENABLE_CUDA__

typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
typedef cudaError_t (*cudaMallocHost_t)(void **, size_t);
typedef cudaError_t (*cudaMallocManaged_t)(void **, size_t, unsigned int);
typedef cudaError_t (*cudaFree_t)(void *);
typedef cudaError_t (*cudaHostFree_t)(void *);

extern "C" cudaError_t cudaMalloc(void **ptr, size_t size) {
  static cudaMalloc_t real_cudaMalloc =
      (cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
  EnableGuard ena;
  cudaError_t result = real_cudaMalloc(ptr, size);
  if (result == cudaSuccess) {
    ena.on(AllocOp::CudaMalloc, *ptr, size, kNone, RETURN_ADDRESS);
  }

  return result;
}

extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
  static cudaMallocHost_t real_cudaMallocHost =
      (cudaMallocHost_t)dlsym(RTLD_NEXT, "cudaMallocHost");
  EnableGuard ena;
  cudaError_t result = real_cudaMallocHost(ptr, size);
  if (result == cudaSuccess) {
    ena.on(AllocOp::CudaHostMalloc, *ptr, size, kNone, RETURN_ADDRESS);
  }

  return result;
}

extern "C" cudaError_t cudaMallocManaged(void **ptr, size_t size,
                                         unsigned int flags) {
  static cudaMallocManaged_t real_cudaMallocManaged =
      (cudaMallocManaged_t)dlsym(RTLD_NEXT, "cudaMallocManaged");
  cudaError_t result = real_cudaMallocManaged(ptr, size, flags);
  EnableGuard ena;
  if (result == cudaSuccess) {
    ena.on(AllocOp::CudaManagedMalloc, *ptr, size, kNone, RETURN_ADDRESS);
  }

  return result;
}

extern "C" cudaError_t cudaFree(void *ptr) {
  static cudaFree_t real_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
  EnableGuard ena;
  cudaError_t result = real_cudaFree(ptr);
  ena.on(AllocOp::CudaFree, ptr, 0, kNone, RETURN_ADDRESS);

  return result;
}

extern "C" cudaError_t cudaHostFree(void *ptr) {
  static cudaHostFree_t real_cudaHostFree =
      (cudaHostFree_t)dlsym(RTLD_NEXT, "cudaFreeHost");
  cudaError_t result = real_cudaHostFree(ptr);

  EnableGuard ena;
  ena.on(AllocOp::CudaHostFree, ptr, 0, kNone, RETURN_ADDRESS);

  return result;
}

#endif