#include<iostream>

//----- Task 1 -----//
// a) Write a CUDA programme that adds the elements of a vector with N elements to each row of a matrix with NxN elements. For this purpose,
//    write a CUDA kernel multi_vector_addition(...) that takes a vector of N doubles and a matrix of NxN doubles as input. You can assume that there are as many threads as elements in the matrix. 
// b) Use CUDA's static shared memory to improve the runtime of the multi_vector_addition(...) kernel. Which data should be stored in shared memory and why? 
// c) Adjust your programme such that the user can specify the size N at runtime. Remember to adjust the grid and block size accordingly.

__global__ void multi_vector_addition(double* vector, double* matrix)
{
	printf("TODO");
}

__global__ void multi_vector_addition_shmem(double* vector, double* matrix)
{
	printf("TODO");
}

__global__ void multi_vector_addition_dynamic_shmem(double* vector, double* matrix)
{
	printf("TODO");
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

// ----- Task 4 -----//
// Use the Nvidia Nsight Compute Profiler to explore the performance characteristics of the code for Task 1.

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

  //----- Task 2 -----//
  int w = f_a(1);
  int x = f_b(2);
  int y = f_c(3);
  int z = f_d(w,x,y);
  std::cout << z << "\n";

  //----- Task 4 -----//

  return EXIT_SUCCESS;

}
