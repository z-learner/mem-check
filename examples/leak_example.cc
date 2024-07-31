
#include <iostream>

#ifdef __ENABLE_CUDA__
#include <cuda_runtime.h>

void *leak_cuda_malloc() {
  void *p;
  cudaMalloc(&p, 100);
  return p;
}

#endif

void *leak_malloc() {
  void *p = malloc(100);
  return p;
}

void *leak_new() {
  void *p = new int[100];
  return p;
}

int main() {
#ifdef __ENABLE_CUDA__
  void *p1 = leak_cuda_malloc();
#endif
  void *p2 = leak_malloc();

  void *p3 = leak_new();

  return 0;
}