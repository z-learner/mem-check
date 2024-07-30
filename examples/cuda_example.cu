#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel(int *d_data) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  d_data[idx] = idx;
}

void testCudaMalloc() {
  int *d_data;
  size_t size = 1024 * sizeof(int);

  cudaMalloc(&d_data, size);
  simpleKernel<<<1, 1024>>>(d_data);
  cudaDeviceSynchronize();
  cudaFree(d_data);
}

void testCudaMallocHost() {
  int *h_data;
  size_t size = 1024 * sizeof(int);

  cudaMallocHost(&h_data, size);
  for (int i = 0; i < 1024; ++i) {
    h_data[i] = i;
  }
  cudaFreeHost(h_data);
}

void testCudaMallocManaged() {
  int *m_data;
  size_t size = 1024 * sizeof(int);

  cudaMallocManaged(&m_data, size);
  simpleKernel<<<1, 1024>>>(m_data);
  cudaDeviceSynchronize();
  cudaFree(m_data);
}

int main() {
  std::cout << "Testing cudaMalloc..." << std::endl;
  testCudaMalloc();

  std::cout << "Testing cudaMallocHost..." << std::endl;
  testCudaMallocHost();

  std::cout << "Testing cudaMallocManaged..." << std::endl;
  testCudaMallocManaged();

  return 0;
}
