#include<iostream>

//----- Task 1 -----//
// Following the CUDA programming guide on the function cudaGetDeviceProperties(cudaDeviceProp* prop, int  device), 
// write a function printDeviceProperties(...) that prints the following GPU (architecture) properties to screen:
// 0. Device name
// 1. clock frequency
// 2. number of streaming multiprocessors
// 3. maximum number of threads per block
// 4. warp size
// ... any other properties that seem interesting to you:-)
// Call the function to explore the GPU at your disposal.
// What do these properties mean for kernel configurations?
void printDeviceProperties()
{
 
}

//----- Task 2 -----//
// a) Write a CUDA kernel that prints the block index and thread index for each thread 
//    in a 1D grid with 1D thread blocks. Compute and print the global index in the grid for each thread.
//
// b) Write a CUDA kernel that prints the block index and thread index for each thread 
//    in a 1D grid with 2D thread blocks. You might want to read the docs on CUDA's dim3 data type to start a kernel with this configuration.//    Compute and print the global index in the grid for each thread.

__global__ void grid1D_block1D() {
 
}

__global__ void grid1D_block2D() {
  
}

//----- Task 3 -----//
// Write a CUDA kernel scalar_mul_strided(...), 
// which multiplies each element of a vector of arbitrary length with a constant.
// One way to achieve this is a grid-strided loop: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

__global__ void scalar_mul_strided(double *v, double *w, double a, const int N) {
 
}

// ----- Task 4 -----//
// Write a CUDA programme that adds two vectors. 
// a) Implement the kernel function as well as all required host- and device-side memory allocations 
// and data transfers.
// b) Use CUDA's unified memory instead of implementing the data transfers manually. 

__global__ void vector_add(double *x, double *y, double *z, const int N) {
  
}

// ----- Task 5 -----//
// Instrument your code with calls to std::chrono to measure the execution times of 
// your compute kernels.
//
// As an example:
//   auto t0 = std::chrono::high_resolution_clock::now();
//   my_kernel<<<...>>>(...);
//   cudaDeviceSynchronize(); //<- Why might this be required?
//   auto t1 = std::chrono::high_resolution_clock::now();
//   std::chrono::duration< double > fs = t1 - t0;
//   std::chrono::milliseconds d = std::chrono::duration_cast< std::chrono::milliseconds >( fs );
//   std::cout << fs.count() << "s\n";
//   std::cout << d.count() << "ms\n";

//----- Code Template -----//
__global__ void scalar_mul(double *v, double *w, double a) {
  //compute global thread id in grid
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  w[tid] = a * v[tid];
}

int main(int argc, char **argv) {
  int N = 10;

  double *a = new double[N];
  double *b = new double[N];
  for (int i = 0; i < N; i++)
    a[i] = 1;

  double *a_d;
  double *b_d;
  cudaMalloc((void **)&a_d, sizeof(double) * N);
  cudaMalloc((void **)&b_d, sizeof(double) * N);
  cudaMemcpy(a_d, a, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(double) * N, cudaMemcpyHostToDevice);

  dim3 numBlocks(1);
  dim3 threadsPerBlock(N);
  scalar_mul<<<numBlocks, threadsPerBlock>>>(a_d, b_d, 5);
  cudaMemcpy(b, b_d, sizeof(double) * N, cudaMemcpyDeviceToHost);
  std::cout << "scalar_mul result:\n";
  for (int i = 0; i < N; i++)
    std::cout << b[i] << " ";
  std::cout << "\n";

  cudaFree(a_d);
  cudaFree(b_d);

  delete[] a;
  delete[] b;

  // Task 1

  // Task 2 a)
 
  // Task 2 b)
  
  // Task 3

  // Task 4 a)
  
  // Task 4 b)
 
  EXIT_SUCCESS;
}
