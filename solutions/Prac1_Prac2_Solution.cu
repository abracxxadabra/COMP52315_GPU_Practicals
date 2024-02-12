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
  int N;
  cudaGetDeviceCount(&N);
  for (int i = 0; i < N; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Device name: " << prop.name << "\n";
    std::cout << "Clock frequency (kHz): " << prop.clockRate << "\n";
    std::cout << "Number of streaming multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "Maximum number of threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Warp size: " << prop.warpSize << "\n";
  }
}

//----- Task 2 -----//
// a) Write a CUDA kernel that prints the block index and thread index for each thread 
//    in a 1D grid with 1D thread blocks. Compute and print the global index in the grid for each thread.
//
// b) Write a CUDA kernel that prints the block index and thread index for each thread 
//    in a 1D grid with 2D thread blocks. You might want to read the docs on CUDA's dim3 data type to start a kernel with this configuration.
//    Compute and print the global index in the grid for each thread.

__global__ void grid1D_block1D() {
  int gid = blockIdx.x *blockDim.x + threadIdx.x;
  printf("thread %d in block %d has global thread id %d \n",threadIdx.x,blockIdx.x,gid);
}

__global__ void grid1D_block2D() {
  int gid = blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x;
  printf("thread (%d, %d) in block %d has global thread id %d \n",threadIdx.x,threadIdx.y,blockIdx.x,gid);
}

//----- Task 3 -----//
// Write a CUDA kernel scalar_mul_strided(...), 
// which multiplies each element of a vector of arbitrary length with a constant.
// One way to achieve this is a grid-strided loop: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

__global__ void scalar_mul_strided(double *v, double *w, double a, const int N) {
  //compute global thread id in grid
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // stride corresponds to grid size -> "grid-strided" loop
  int stride = gridDim.x * blockDim.x;
  for(int i = tid; i < N;i += stride)
  {
    w[i] = a * v[i];
  }
}

// ----- Task 4 -----//
// Write a CUDA programme that adds two vectors. 
// a) Implement the kernel function as well as all required host- and device-side memory allocations 
// and data transfers.
// b) Use CUDA's unified memory instead of implementing the data transfers manually. 

__global__ void vector_add(double *x, double *y, double *z, const int N) {
  //compute global thread id in grid
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // stride corresponds to block size -> "block-strided" loop
  int stride = blockDim.x;
  for(unsigned int i = tid; i < N;i += stride)
  {
    z[i] = x[i] + y[i];
  }
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

  // Task 1
  std::cout << "Device properties:\n";
  printDeviceProperties();

  // Task 2 a)
  dim3 block1D(2);
  dim3 grid1D(4);

  std::cout << "1D grid with 1D blocks:\n";
  grid1D_block1D<<<grid1D,block1D>>>();

  // Task 2 b)
  std::cout << "1D grid with 2D blocks:\n";
  dim3 block2D(3);
  grid1D_block2D<<<grid1D, block2D>>>();


  // Task 3
  int M = 37;
  double *v = new double[M];
  double *w = new double[M];
  for (int i = 0; i < M; i++)
    v[i] = 1;

  double *v_d;
  double *w_d;
  for (int i = 0; i < M; i++)
    w[i] = 0;
  cudaMalloc((void **)&v_d, sizeof(double) * M);
  cudaMalloc((void **)&w_d, sizeof(double) * M);
  cudaMemcpy(v_d, v, sizeof(double) * M, cudaMemcpyHostToDevice);
  cudaMemcpy(w_d, w, sizeof(double) * M, cudaMemcpyHostToDevice);
  scalar_mul_strided<<<4,2>>>(v_d, w_d, 5, M);
  cudaMemcpy(w, w_d, sizeof(double) * M, cudaMemcpyDeviceToHost);
  std::cout << "scalar_mul_strided result:\n";
  for (int i = 0; i < M; i++)
    std::cout << w[i] << " ";
  std::cout << "\n";

  // Task 4 a)
  double *x = new double[M];
  double *y = new double[M];
  double *z = new double[M];

  for (int i = 0; i < M; i++)
  {
    x[i] = 1;
    y[i] = 1;
  }

  double *x_d;
  double *y_d;
  double *z_d;
  cudaMalloc((void **)&x_d, sizeof(double) * M);
  cudaMalloc((void **)&y_d, sizeof(double) * M);
  cudaMalloc((void **)&z_d, sizeof(double) * M);
  cudaMemcpy(x_d, x, sizeof(double) * M, cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, sizeof(double) * M, cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z, sizeof(double) * M, cudaMemcpyHostToDevice);
  vector_add<<<4, 2>>>(x_d,y_d,z_d,M);
  cudaMemcpy(z, z_d, sizeof(double) * M, cudaMemcpyDeviceToHost);

  std::cout << "vector_add result:\n";
  for (int i = 0; i < M; i++)
    std::cout << z[i] << " ";
  std::cout << "\n";

  // Task 4 b)
  double *c, *d, *e;

  cudaMallocManaged(&c, sizeof(double)*M);
  cudaMallocManaged(&d, sizeof(double)*M);
  cudaMallocManaged(&e, sizeof(double)*M);

  for (int i = 0; i < M; i++)
  {
    c[i] = 3;
    d[i] = 1;
  }

  vector_add<<<4, 2>>>(c,d,e,M);
  cudaDeviceSynchronize();

  std::cout << "vector_add result:\n";
  for (int i = 0; i < M; i++)
    std::cout << e[i] << " ";
  std::cout << "\n";

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(v_d);
  cudaFree(w_d);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  delete[] a;
  delete[] b;
  delete[] v;
  delete[] w;
  delete[] x;
  delete[] y;
  delete[] z;

  EXIT_SUCCESS;
}
