#pragma once

#include <cstddef>
#include <cstdint>
#include <map>

enum class AllocOp {
  New,
  Delete,
  NewArray,
  DeleteArray,
  Malloc,
  Free,
  CudaHostMalloc,
  CudaMalloc,
  CudaManagedMalloc,
  CudaFree,
  CudaHostFree,
  Unknown,
};

struct AllocAction {
  AllocOp op;
  uint32_t tid;
  void *ptr;
  size_t size;
  size_t align;
  void *caller;
  int64_t timestamp;
};

constexpr const char *kAllocOpNames[] = {
    "New",
    "Delete",
    "NewArray",
    "DeleteArray",
    "Malloc",
    "Free",
    "CudaHostMalloc",
    "CudaDeviceMalloc",
    "CudaManagedMalloc",
    "CudaFree",
    "CudaHostFree",
    "Unknown",
};

constexpr bool kAllocOpIsAllocation[] = {
    true, false, true, false, true,  false,
    true, true,  true, false, false, false,
};

constexpr bool kAllocOpIsCpp[] = {
    true,  true,  true,  true,  false, false,
    false, false, false, false, false, false,
};

constexpr bool kAllocOpIsC[] = {
    false, false, false, false, true,  true,
    false, false, false, false, false, false,
};

constexpr bool kAllocOpIsCuda[] = {
    false, false, false, false, false, false,
    true,  true,  true,  true,  true,  false,
};

constexpr AllocOp kAllocOpFreeFunction[] = {
    AllocOp::Delete,       AllocOp::Unknown,  AllocOp::DeleteArray,
    AllocOp::Unknown,      AllocOp::Free,     AllocOp::Unknown,
    AllocOp::CudaHostFree, AllocOp::CudaFree, AllocOp::CudaFree,
    AllocOp::Unknown,      AllocOp::Unknown,  AllocOp::Unknown,
};

constexpr size_t kNone = static_cast<size_t>(-1);