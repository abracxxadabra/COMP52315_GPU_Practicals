#include <iostream>

#define THREADS_PER_BLOCK 256
#define BLOCKS 10

__global__ void kernelA(int *a, int n) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadId; i < n; i += gridDim.x * blockDim.x) {
    a[i] = a[i] + 1;
  }
}

__global__ void kernelB(int *b, int n) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadId; i < n; i += gridDim.x * blockDim.x) {
    b[i] = b[i] * 2;
  }
}

int main() {
  long long N = 1000000;
  int *a, *b;
  cudaMallocManaged(&a, N * sizeof(int));
  cudaMallocManaged(&b, N * sizeof(int));

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  kernelA<<<BLOCKS,THREADS_PER_BLOCK, 0, stream1>>>(a, N);
  kernelB<<<BLOCKS, THREADS_PER_BLOCK, 0, stream2>>>(b, N);
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  for (int i = 0; i < 10; i++) {
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < 10; i++) {
    std::cout << b[i] << " ";
  }
  std::cout << std::endl;

  cudaFree(a);
  cudaFree(b);

  EXIT_SUCCESS;
}
