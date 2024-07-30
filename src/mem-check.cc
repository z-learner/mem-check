#include "addr2sym.hpp"
#include "alloc-action.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <new>

namespace detail {

struct GlobalData {
  std::map<void *, AllocAction> allocated{};
  std::mutex lock{};
  bool enable{false};

  GlobalData() : enable(true) {}

  ~GlobalData() {
    enable = false;
    for (auto &&[ptr, info] : allocated) {
      printf("检测到内存泄漏 ptr = %p, size = %zd, caller = %s\n", ptr,
             info.size, addr2sym(info.caller).c_str());
    }
  }

  void on(AllocOp op, void *ptr, size_t size, size_t align, void *caller) {
    if (!ptr) {
      return;
    }
    // printf("%s(ptr=%p, size=%zd, align=%zd, caller=%p)\n",
    // kAllocOpNames[(size_t)op], ptr, size, align, caller);
    std::lock_guard<std::mutex> guard(lock);
    if (kAllocOpIsAllocation[static_cast<size_t>(op)]) {
      auto result = allocated.insert(
          {ptr, AllocAction{op, 0, ptr, size, align, caller, 0}});
      if (!result.second) {
        printf("检测到内存多次分配同一个地址 ptr = %p, size = %zd, "
               "caller = %s\n",
               ptr, size, addr2sym(caller).c_str());
      }
    } else {
      auto it = allocated.find(ptr);
      if (it == allocated.end()) {
        printf("检测到尝试释放不存在的内存 ptr = %p, size = %zd, "
               "caller = %s\n",
               ptr, size, addr2sym(caller).c_str());
        return;
      }

      const AllocAction &action = it->second;
      bool wrong_free_func =
          kAllocOpFreeFunction[static_cast<size_t>(action.op)] != op;
      bool wrong_size = (size != kNone && action.size != size);
      bool wrong_align = (align != kNone && action.align != align);

      if (wrong_free_func || wrong_size || wrong_align) {
        printf("检测到内存释放错误 ptr = %p, size = %zd, caller = %s\n", ptr,
               action.size, addr2sym(caller).c_str());
        if (wrong_free_func) {
          printf("使用了错误的释放函数\n");
        }
        if (wrong_size) {
          printf("指定了错误的大小 size = %zd, 正确大小 = %zd\n", size,
                 action.size);
        }
        if (wrong_align) {
          printf("指定了错误的对齐 align = %zd, 正确对齐 = %zd\n", align,
                 action.align);
        }
      }
      allocated.erase(it);
    }
  }
} global;

struct EnableGuard {
  bool was_enable{};

  EnableGuard() {
    was_enable = global.enable;
    global.enable = false;
  }

  explicit operator bool() { return was_enable; }

  ~EnableGuard() { global.enable = was_enable; }
};

} // namespace detail

void operator delete(void *ptr) noexcept(false) {
  detail::EnableGuard ena;

  if (ena) {
    detail::global.on(AllocOp::Delete, ptr, kNone, kNone,
                      __builtin_return_address(0));
  }
  free(ptr);
}

void operator delete[](void *ptr) noexcept(false) {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::DeleteArray, ptr, kNone, kNone,
                      __builtin_return_address(0));
  }
  free(ptr);
}

void *operator new(size_t size) noexcept(false) {
  detail::EnableGuard ena;
  void *ptr = malloc(size);
  if (ena) {
    detail::global.on(AllocOp::New, ptr, size, kNone,
                      __builtin_return_address(0));
  }
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new[](size_t size) noexcept(false) {
  detail::EnableGuard ena;
  void *ptr = malloc(size);
  if (ena) {
    detail::global.on(AllocOp::NewArray, ptr, size, kNone,
                      __builtin_return_address(0));
  }
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new(size_t size, std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  void *ptr = malloc(size);
  if (ena) {
    detail::global.on(AllocOp::New, ptr, size, kNone,
                      __builtin_return_address(0));
  }
  return ptr;
}

void *operator new[](size_t size, std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  void *ptr = malloc(size);
  if (ena) {
    detail::global.on(AllocOp::NewArray, ptr, size, kNone,
                      __builtin_return_address(0));
  }
  return ptr;
}

void operator delete(void *ptr, size_t size) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::Delete, ptr, size, kNone,
                      __builtin_return_address(0));
  }
  free(ptr);
}

void operator delete[](void *ptr, size_t size) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::DeleteArray, ptr, size, kNone,
                      __builtin_return_address(0));
  }
  free(ptr);
}

void operator delete(void *ptr, std::align_val_t align) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::Delete, ptr, kNone, static_cast<size_t>(align),
                      __builtin_return_address(0));
  }
  free(ptr);
}

void operator delete[](void *ptr, std::align_val_t align) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::DeleteArray, ptr, kNone,
                      static_cast<size_t>(align), __builtin_return_address(0));
  }
  free(ptr);
}

void operator delete(void *ptr, std::size_t size,
                     std::align_val_t align) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::Delete, ptr, size, static_cast<size_t>(align),

                      __builtin_return_address(0));
  }
  free(ptr);
}

void operator delete[](void *ptr, std::size_t size,
                       std::align_val_t align) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::DeleteArray, ptr, size,
                      static_cast<size_t>(align),

                      __builtin_return_address(0));
  }
  free(ptr);
}

void operator delete(void *ptr, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::Delete, ptr, kNone, static_cast<size_t>(align),
                      __builtin_return_address(0));
  }
  free(ptr);
}

void operator delete[](void *ptr, std::align_val_t align,
                       std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  if (ena) {
    detail::global.on(AllocOp::DeleteArray, ptr, kNone,
                      static_cast<size_t>(align), __builtin_return_address(0));
  }
  free(ptr);
}

void *operator new(std::size_t size, std::align_val_t align) noexcept(false) {
  detail::EnableGuard ena;
  void *ptr = aligned_alloc(static_cast<size_t>(align), size);
  if (ena) {
    detail::global.on(AllocOp::New, ptr, size, static_cast<size_t>(align),
                      __builtin_return_address(0));
  }
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new[](size_t size, std::align_val_t align) noexcept(false) {
  detail::EnableGuard ena;
  void *ptr = aligned_alloc(static_cast<size_t>(align), size);
  if (ena) {
    detail::global.on(AllocOp::NewArray, ptr, size, static_cast<size_t>(align),
                      __builtin_return_address(0));
  }
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void *operator new(std::size_t size, std::align_val_t align,
                   std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  void *ptr = aligned_alloc(static_cast<size_t>(align), size);
  if (ena) {
    detail::global.on(AllocOp::New, ptr, size, static_cast<size_t>(align),
                      __builtin_return_address(0));
  }
  return ptr;
}

void *operator new[](size_t size, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
  detail::EnableGuard ena;
  void *ptr = aligned_alloc(static_cast<size_t>(align), size);
  if (ena) {
    detail::global.on(AllocOp::NewArray, ptr, size, static_cast<size_t>(align),
                      __builtin_return_address(0));
  }
  return ptr;
}