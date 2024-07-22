#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

enum class AllocOp : uint8_t {
  Unknow,
  New,
  Delete,
  NewArray,
  DeleteArray,
  Malloc,
  Free
}; // enum class AllocOp

struct AllocAction {
  AllocOp op;
  uint32_t tid;
  void *ptr;
  size_t size;
  size_t align;
  void *caller;
  uint64_t timestamp;
};

const static std::unordered_map<AllocOp, const char *> kAllocOp2Name = {
    {AllocOp::Unknow, "Unknow"},
    {AllocOp::New, "New"},
    {AllocOp::Delete, "Delete"},
    {AllocOp::NewArray, "NewArray"},
    {AllocOp::DeleteArray, "DeleteArray"},
    {AllocOp::Malloc, "Malloc"},
    {AllocOp::Free, "Free"}};

const static std::unordered_map<AllocOp, bool> kAllocOpIsAllocation = {
    {AllocOp::Unknow, false},      {AllocOp::New, true},
    {AllocOp::Delete, false},      {AllocOp::NewArray, true},
    {AllocOp::DeleteArray, false}, {AllocOp::Malloc, true},
    {AllocOp::Free, false}};

const static std::unordered_map<AllocOp, bool> kAllocOpIsC = {
    {AllocOp::Unknow, false},     {AllocOp::New, false},
    {AllocOp::Delete, false},      {AllocOp::NewArray, false},
    {AllocOp::DeleteArray, false}, {AllocOp::Malloc, true},
    {AllocOp::Free, true}};

const static std::unordered_map<AllocOp, bool> kAllocOpIsCpp = {
    {AllocOp::Unknow, false},     {AllocOp::New, true},
    {AllocOp::Delete, true},      {AllocOp::NewArray, true},
    {AllocOp::DeleteArray, true}, {AllocOp::Malloc, false},
    {AllocOp::Free, false}};


const static std::unordered_map<AllocOp, AllocOp> kAllocOp2FreeAction = {
    {AllocOp::Unknow, AllocOp::Unknow},
    {AllocOp::New, AllocOp::Delete},
    {AllocOp::Delete, AllocOp::Unknow},
    {AllocOp::NewArray, AllocOp::DeleteArray},
    {AllocOp::DeleteArray, AllocOp::Unknow},
    {AllocOp::Malloc, AllocOp::Free},
    {AllocOp::Free, AllocOp::Unknow}};


