#include<iostream>

#define N 10
#define M 10

//----- Task 1 -----//
// a) Write a CUDA programme that adds the elements of a vector with N elements to each row of a matrix with NxN elements. For this purpose,
//    write a CUDA kernel multi_vector_addition(...) that takes a vector of N doubles and a matrix of NxN doubles as input. You can assume that there are as many threads as elements in the matrix. 
// b) Use CUDA's static shared memory to improve the runtime of the multi_vector_addition(...) kernel. Which data should be stored in shared memory and why? 
// c) Adjust your programme such that the user can specify the size N at runtime. Remember to adjust the grid and block size accordingly.

__global__ void multi_vector_addition(double* vector, double* matrix)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x *blockDim.x + threadIdx.x;
	matrix[gid] = matrix[gid] + vector[tid];
}

__global__ void multi_vector_addition_shmem(double* vector, double* matrix)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x *blockDim.x + threadIdx.x;
	__shared__ double v[N];
	v[tid] = vector[tid];
	__syncthreads();
	matrix[gid] = matrix[gid] + v[tid];
}

//----- Task 2 -----//
//   Consider the subsequent functions f_a(...), f_b(...), f_c(...), f_d(...) and the data dependencies between their parameters as evident in the main(...) function.
// a) Draw the task graph that outlines the dependencies between these functions. Each node in the graph is either a function name, or a parameter. Edges are directed
//    and represent input and output dependencies. Input dependencies 
// b) Rewrite the functions f_a(...), f_b(...), f_c(...), f_d(...) as CUDA kernels.
// c) Complying to data dependencies, launch the kernels concurrently by using CUDA streams.

int f_a(const int a) {
  return a+1;
}

int f_b(const int b) {
  return b+1;
}

int f_c(const int c) {
  return c+1;
}

int f_d(const int a, const int b, const int c) {
  return a+b+c;
}

__global__ void kernel_f_a(int* a, int* res_a) {
  *res_a = *a+1;
}

__global__ void kernel_f_b(int* b, int* res_b) {
  *res_b = *b+1;
}

__global__ void kernel_f_c(int* c, int* res_c) {
  *res_c = *c+1;
}

__global__ void kernel_f_d(int* a, int* b, int* c, int* res_d) {
  *res_d = *a+*b+*c;
}

// ----- Task 3 -----//
// Instrument your code with calls to std::chrono to measure the execution times of 
// your kernels.
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

int main(int argc, char **argv) {
  //----- Task 1 -----//
  // Uncomment to activate helper code to retrieve N as commandline parameter in Task 1 c):
  // int N;
  // if (argc == 2)
  // {
  //  N = std::stoi(argv[1]);
  // } else 
  // {
  // std::cout << "Error: Missing problem size N. Please provide N as "
  //               "commandline parameter."
  //            << std::endl;
  //  exit(0);
  // }
  double *v, *m;
  cudaMallocManaged(&v, sizeof(double)*N);
  cudaMallocManaged(&m, sizeof(double)*N*N);

  for (int i = 0; i < N; i++)
  {
    v[i] = 1;
  }

  for (int i = 0; i < N*N; i++)
  {
    m[i] = 1;
  }

  // a) 
  multi_vector_addition<<<10,10>>>(v,m);
  cudaDeviceSynchronize();

  for (int i = 0; i < N*N; i++)
    std::cout << m[i] << " ";
  std::cout << "\n";

  // b) 
  multi_vector_addition_shmem<<<10,10>>>(v,m);
  cudaDeviceSynchronize();

  for (int i = 0; i < N*N; i++)
    std::cout << m[i] << " ";
  std::cout << "\n";

  cudaFree(v);
  cudaFree(m);

  //----- Task 2 -----//
  int w = f_a(1);
  int x = f_b(2);
  int y = f_c(3);
  int z = f_d(w,x,y);
  std::cout << z << "\n";

  int *a, *b, *c, *d, *res_a, *res_b, *res_c, *res_d;
  cudaMallocManaged(&a, sizeof(int));
  cudaMallocManaged(&b, sizeof(int));
  cudaMallocManaged(&c, sizeof(int));
  cudaMallocManaged(&d, sizeof(int));
  cudaMallocManaged(&res_a, sizeof(int));
  cudaMallocManaged(&res_b, sizeof(int));
  cudaMallocManaged(&res_c, sizeof(int));
  cudaMallocManaged(&res_d, sizeof(int));
  *a = 1;
  *b = 2;
  *c = 3;

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  kernel_f_a<<<1, 1, 0, stream1>>>(a, res_a);
  kernel_f_b<<<1, 1, 0, stream2>>>(b, res_b);
  kernel_f_c<<<1, 1, 0, stream3>>>(c, res_c);
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  cudaStreamSynchronize(stream3);
  kernel_f_d<<<1, 1, 0>>>(res_a,res_b,res_c,res_d);
  cudaDeviceSynchronize();
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);

  std::cout << *res_d << "\n";
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(d);
  cudaFree(res_a);
  cudaFree(res_b);
  cudaFree(res_c);
  cudaFree(res_d);

  //----- Task 4 -----//

  return EXIT_SUCCESS;

}
