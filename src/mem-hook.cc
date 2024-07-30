#include "addr2sym.hpp"
#include "plot-actions.hpp"

#include "alloc-action.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fstream>

#include <mutex>
#include <new>

#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <memory_resource>

namespace detail {

static uint32_t get_thread_id() { return gettid(); }

struct alignas(64) PerThreadData {
  size_t const buff_size{64 * 1024 * 1024};
  void *buff = mmap(nullptr, buff_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  std::pmr::monotonic_buffer_resource mono{buff, buff_size};
  std::pmr::unsynchronized_pool_resource pool{&mono};
  std::recursive_mutex lock;
  std::pmr::deque<AllocAction> actions{&pool};
  bool enable{false};
};

struct GlobalData {
  std::mutex lock{};
  static constexpr size_t kPreThreadCount{8};
  std::array<PerThreadData, kPreThreadCount> per_threads{};
  bool export_plot_on_exit{true};

  std::thread export_thread;
  std::atomic<bool> stopped{false};

  GlobalData() {
    for (size_t index = 0; index < kPreThreadCount; ++index) {
      per_threads[index].enable = true;
    }
  }

  PerThreadData &get_per_thread_data(uint32_t tid) {
    // why multiply by 17?
    return per_threads[static_cast<size_t>(tid) * 17 % kPreThreadCount];
  }

  void export_thread_entry(const std::string &filename) {
    size_t const buff_size = 64 * 1024 * 1024;
    void *buff = mmap(nullptr, buff_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    std::pmr::monotonic_buffer_resource mono{buff, buff_size};
    std::pmr::unsynchronized_pool_resource pool{&mono};
    std::ofstream ofs{filename};
    std::pmr::deque<AllocAction> actions;
    // save on running
    while (!stopped.load(std::memory_order_relaxed)) {
      for (auto &per_thread : per_threads) {
        std::unique_lock<std::recursive_mutex> guard(per_thread.lock);
        auto thread_actions = std::move(per_thread.actions);
        guard.unlock();
        actions.insert(actions.end(), thread_actions.begin(),
                       thread_actions.end());
      }
      for (auto &action : actions) {
        // write binary data to file
        ofs.write(reinterpret_cast<char *>(&action), sizeof(AllocAction));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // save after running
    for (auto &per_thread : per_threads) {
      std::unique_lock<std::recursive_mutex> guard(per_thread.lock);
      auto thread_actions = std::move(per_thread.actions);
      guard.unlock();
      actions.insert(actions.end(), thread_actions.begin(),
                     thread_actions.end());
    }
    for (auto &action : actions) {
      // write binary data to file
      ofs.write(reinterpret_cast<char *>(&action), sizeof(AllocAction));
    }
  }

  ~GlobalData() {
    for (auto &per_thread : per_threads) {
      per_thread.enable = false;
    }

    if (export_thread.joinable()) {
      stopped.store(true, std::memory_order_relaxed);
      export_thread.join();
    }

    if (export_plot_on_exit) {
      std::vector<AllocAction> actions;
      for (auto &per_thread : per_threads) {
        auto &their_actions = per_thread.actions;
        actions.insert(actions.end(), their_actions.begin(),
                       their_actions.end());
      }
      mem_check_plot_alloc_actions(std::move(actions));
    }
  }
};

GlobalData *global = nullptr;

struct EnableGuard {
  uint32_t tid;
  bool was_enable;
  PerThreadData *per_thread;
  EnableGuard()
      : tid(get_thread_id()),
        per_thread(global ? &global->get_per_thread_data(tid) : nullptr) {
    if (!per_thread) {
      was_enable = false;
    } else {
      per_thread->lock.lock();
      was_enable = per_thread->enable;
      per_thread->lock.unlock();
    }
  }

  explicit operator bool() const { return was_enable; }

  void on(AllocOp op, void *ptr, size_t size, size_t align, void *caller) {
    if (ptr) {
      auto now = std::chrono::high_resolution_clock::now();
      int64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              now.time_since_epoch())
                              .count();
      per_thread->lock.lock();
      per_thread->actions.emplace_back(
          AllocAction{op, tid, ptr, size, align, caller, timestamp});
      per_thread->lock.unlock();
    }
  }

  ~EnableGuard() {
    if (per_thread) {
      per_thread->lock.lock();
      per_thread->enable = was_enable;
      per_thread->lock.unlock();
    }
  }
};

} // namespace detail

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

extern "C" void *malloc(size_t size) noexcept {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(malloc)(size);
  if (ena) {
    ena.on(AllocOp::Malloc, ptr, size, kNone, RETURN_ADDRESS);
  }
  return ptr;
}

extern "C" void free(void *ptr) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::Free, ptr, 0, kNone, RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

extern "C" void *calloc(size_t nmemb, size_t size) noexcept {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(calloc)(nmemb, size);
  if (ena) {
    ena.on(AllocOp::Malloc, ptr, nmemb * size, kNone, RETURN_ADDRESS);
  }
  return ptr;
}

extern "C" void *realloc(void *ptr, size_t size) noexcept {
  detail::EnableGuard ena;
  void *new_ptr = REAL_LIBC(realloc)(ptr, size);
  if (ena) {
    ena.on(AllocOp::Malloc, new_ptr, size, kNone, RETURN_ADDRESS);
  }
  return new_ptr;
}

extern "C" void *reallocarray(void *ptr, size_t nmemb, size_t size) noexcept {
  detail::EnableGuard ena;
  void *new_ptr = REAL_LIBC(reallocarray)(ptr, nmemb, size);
  if (ena) {
    ena.on(AllocOp::Malloc, new_ptr, nmemb * size, kNone, RETURN_ADDRESS);
  }
  return new_ptr;
}

extern "C" void *valloc(size_t size) noexcept {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(valloc)(size);
  size_t pagesize = sysconf(_SC_PAGESIZE);
  if (ena) {
    ena.on(AllocOp::Malloc, ptr, size, pagesize, RETURN_ADDRESS);
  }
  return ptr;
}

extern "C" void *memalign(size_t alignment, size_t size) noexcept {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(memalign)(alignment, size);
  if (ena) {
    ena.on(AllocOp::Malloc, ptr, size, alignment, RETURN_ADDRESS);
  }
  return ptr;
}

extern "C" int posix_memalign(void **memptr, size_t alignment, size_t size) {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(memalign)(alignment, size);
  if (ena) {
    ena.on(AllocOp::Malloc, *memptr, size, alignment, RETURN_ADDRESS);
  }
  int ret = 0;
  if (!ptr) {
    ret = errno;
  } else {
    *memptr = ptr;
  }
  return ret;
}

void operator delete(void *ptr) noexcept(false) {
  detail::EnableGuard ena;

  if (ena) {
    ena.on(AllocOp::Delete, ptr, kNone, kNone, RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr) noexcept(false) {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::DeleteArray, ptr, kNone, kNone, RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void *operator new(size_t size) noexcept(false) {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(malloc)(size);
  if (ena) {
    ena.on(AllocOp::New, ptr, size, kNone, RETURN_ADDRESS);
  }
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new[](size_t size) noexcept(false) {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(malloc)(size);
  if (ena) {
    ena.on(AllocOp::NewArray, ptr, size, kNone, RETURN_ADDRESS);
  }
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new(size_t size, std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(malloc)(size);
  if (ena) {
    ena.on(AllocOp::New, ptr, size, kNone, RETURN_ADDRESS);
  }
  return ptr;
}

void *operator new[](size_t size, std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(malloc)(size);
  if (ena) {
    ena.on(AllocOp::NewArray, ptr, size, kNone, RETURN_ADDRESS);
  }
  return ptr;
}

void operator delete(void *ptr, size_t size) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::Delete, ptr, size, kNone, RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, size_t size) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::DeleteArray, ptr, size, kNone, RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void operator delete(void *ptr, std::align_val_t align) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::Delete, ptr, kNone, static_cast<size_t>(align),
           RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, std::align_val_t align) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::DeleteArray, ptr, kNone, static_cast<size_t>(align),
           RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void operator delete(void *ptr, std::size_t size,
                     std::align_val_t align) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::Delete, ptr, size, static_cast<size_t>(align),

           RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, std::size_t size,
                       std::align_val_t align) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::DeleteArray, ptr, size, static_cast<size_t>(align),

           RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void operator delete(void *ptr, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::Delete, ptr, kNone, static_cast<size_t>(align),
           RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, std::align_val_t align,
                       std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    ena.on(AllocOp::DeleteArray, ptr, kNone, static_cast<size_t>(align),
           RETURN_ADDRESS);
  }
  REAL_LIBC(free)(ptr);
}

void *operator new(std::size_t size, std::align_val_t align) noexcept(false) {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(memalign)(static_cast<size_t>(align), size);
  if (ena) {
    ena.on(AllocOp::New, ptr, size, static_cast<size_t>(align), RETURN_ADDRESS);
  }
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new[](size_t size, std::align_val_t align) noexcept(false) {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(memalign)(static_cast<size_t>(align), size);
  if (ena) {
    ena.on(AllocOp::NewArray, ptr, size, static_cast<size_t>(align),
           RETURN_ADDRESS);
  }
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new(std::size_t size, std::align_val_t align,
                   std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(memalign)(static_cast<size_t>(align), size);
  if (ena) {
    ena.on(AllocOp::New, ptr, size, static_cast<size_t>(align), RETURN_ADDRESS);
  }
  return ptr;
}

void *operator new[](size_t size, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  void *ptr = REAL_LIBC(memalign)(static_cast<size_t>(align), size);
  if (ena) {
    ena.on(AllocOp::NewArray, ptr, size, static_cast<size_t>(align),
           RETURN_ADDRESS);
  }
  return ptr;
}

static detail::GlobalData global_buff;
static int global_init_helper = (detail::global = &global_buff, 0);