#include <chrono>
#include <iostream>
#include <omp.h>

#define N 20000

//----- Task 1 -----//
// a) Write an OpenMP CPU programme that adds the elements of a vector with N
//    elements to each row of a matrix with NxN elements. For this purpose,
//    write a function multi_vector_addition_CPU(...) that takes a vector of N
//    doubles and a matrix of NxN doubles as input.
// b) Use OpenMP's target directive to write a function
//    multi_vector_addition_GPU(...) that parallelizes this functionality on GPUs.
//    Ensure that the workload is distributed among teams *and* among threads in a
//    team. 
// c) Instrument your code with calls to std::chrono to measure the
//    execution times of your functions.
//    As an example:
//     auto t0 = std::chrono::high_resolution_clock::now();
//     my_function(...);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     std::chrono::duration< double > duration = t1 - t0;
//     std::chrono::milliseconds ms_duration = std::chrono::duration_cast<
//     std::chrono::milliseconds >( duration ); std::cout << duration.count() <<
//     "s\n"; std::cout << ms_duration.count() << "ms\n";
// d) Which version of the function runs faster? What could be the reason for
//    this?

void multi_vector_addition_CPU(double *vector, double *matrix) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrix[j + i * N] = matrix[j + i * N] + vector[j];
    }
  }
}

void multi_vector_addition_GPU(double *vector, double *matrix) {
#pragma omp target teams distribute parallel for map(to: vector [0:N]) map(tofrom: matrix [0:N * N])
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrix[j + i * N] = matrix[j + i * N] + vector[j];
    }
  }
}

//----- Task 2 -----//
//   Consider the subsequent functions f_a(...), f_b(...), f_c(...), f_d(...)
//   and the data dependencies between their parameters as evident in the
//   main(...) function. 
//   a) Enable the concurrent execution of the functions on
//      the CPU via the OpenMP task depend clause to comply to data dependencies.
//   b) Adjust your code such that the functions f_a, f_b and f_c can be
//      executed concurrently on the GPU. Make use of OpenMP's reduction clause.
void f_a(double *a, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += a[i];
  }
  *res = acc;
}

void f_b(double *b, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += b[i];
  }
  *res = acc;
}

void f_c(double *c, double *res) {
  double acc = 0;
  for (int i = 0; i < N; i++) {
    acc += c[i];
  }
  *res = acc;
}

void f_d(double a, double b, double c, double *res) { *res = a + b + c; }

void f_a_gpu(double *a, double *res) {
  double acc = 0;
#pragma omp target teams distribute parallel for reduction(+:acc) map(to: a[0:N]) map(tofrom: acc)
  for (int i = 0; i < N; i++) {
    acc += a[i];
  }
  *res = acc;
}

void f_b_gpu(double *b, double *res) {
  double acc = 0;
#pragma omp target teams distribute parallel for reduction(+:acc) map(to: b[0:N]) map(tofrom: acc)
  for (int i = 0; i < N; i++) {
    acc += b[i];
  }
  *res = acc;
}

void f_c_gpu(double *c, double *res) {
  double acc = 0;
#pragma omp target teams distribute parallel for reduction(+:acc) map(to: c[0:N]) map(tofrom: acc)
  for (int i = 0; i < N; i++) {
    acc += c[i];
  }
  *res = acc;
}

//----- Code Template -----//

int main(int argc, char **argv) {
  //----- Task 1 -----//
  // Uncomment to activate helper code to retrieve N as commandline parameter in
  // Task 1 c): int N; if (argc == 2)
  // {
  //  N = std::stoi(argv[1]);
  // } else
  // {
  // std::cout << "Error: Missing problem size N. Please provide N as "
  //               "commandline parameter."
  //            << std::endl;
  //  exit(0);
  // }

  double *vector = new double[N];
  double *matrix = new double[N * N];

  for (int i = 0; i < N; i++) {
    vector[i] = 1;
  }

  for (int i = 0; i < N * N; i++) {
    matrix[i] = 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  multi_vector_addition_CPU(vector, matrix);
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "OpenMP CPU result: \n";
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  std::chrono::duration<double> duration = t1 - t0;
  std::chrono::milliseconds ms_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  std::cout << ms_duration.count() << "ms\n";

  t0 = std::chrono::high_resolution_clock::now();
  multi_vector_addition_GPU(vector, matrix);
  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "OpenMP GPU result: \n";
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  duration = t1 - t0;
  ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  std::cout << ms_duration.count() << "ms\n";

  //----- Task 2 -----//
  // a)
  double *a = new double[N];
  double *b = new double[N];
  double *c = new double[N];
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 1;
    c[i] = 1;
  }
  double w, x, y, z;
#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp task depend(out : w) depend(in : a)
      { f_a(a, &w); }

#pragma omp task depend(out : x) depend(in : b)
      { f_b(b, &x); }

#pragma omp task depend(out : y) depend(in : c)
      { f_c(c, &y); }

#pragma omp task depend(in : w, x, y) depend(out : z)
      {
        f_d(w, x, y, &z);
        std::cout << "The value of z is " << z << "."
                  << "\n";
      }
    }
  }

  //----- Task 2 -----//
  // b)
#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp task depend(out : w) depend(in : a)
      { f_a_gpu(a, &w); }

#pragma omp task depend(out : x) depend(in : b)
      { f_b_gpu(b, &x); }

#pragma omp task depend(out : y) depend(in : c)
      { f_c_gpu(c, &y); }

#pragma omp task depend(in : w, x, y) depend(out : z)
      {
        f_d(w, x, y, &z);
        std::cout << "The value of z is " << z << "."
                  << "\n";
      }
    }
  }

  return EXIT_SUCCESS;
}
