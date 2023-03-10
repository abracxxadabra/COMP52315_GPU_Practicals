#include<iostream>
#include<vector>
#include<sycl/sycl.hpp>

int main(int argc, char **argv) 
{
  unsigned long N;
  if (argc == 2)
  {
   N = std::stoi(argv[1]);
  } else 
  {
  std::cout << "Error: Missing problem size N. Please provide N as commandline parameter."
            << std::endl;
   exit(0);
  }

  
  //----- Task 1 -----//
  // Use SYCL to query device information.

  //----- Task 2 -----//
  // a) Write a SYCL programme that implements the element-wise addition of two vectors 
  //    by using the high-level version of the parallel_for kernel invocation API.
  // b) Write a SYCL programme that implements the element-wise addition of two vectors 
  //    by using the ND range version of the parallel_for kernel invocation API.
  // c) Execute both programmes once on a CPU, and once on a GPU, by constructing a SYCL queue 
  //    with the corresponding device selectors. 

  std::vector<double> x(N);
  std::vector<double> y(N);
  std::vector<double> z(N);

  for(auto& i : x) i = 1.0;
  for(auto& i : y) i = 1.0;
  for(auto& i : z) i = 0.0;

  sycl::queue q;


  // a)
  {
  sycl::buffer<double,1> x_buf(x.data(), x.size());
  sycl::buffer<double,1> y_buf(y.data(), y.size());
  sycl::buffer<double,1> z_buf(z.data(), z.size());
  q.submit([&](sycl::handler& cgh)
  {
      auto x_d = x_buf.get_access<sycl::access::mode::read>(cgh);
      auto y_d = y_buf.get_access<sycl::access::mode::read>(cgh);
      auto z_d = z_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
      		     z_d[i] = x_d[i] + y_d[i];
      });

  });
  }

  std::cout << "Vector addition result: \n";
  for(auto i : z) std::cout << i << " ";
  std::cout << "\n";

  // b)
  sycl::range global{N};
  sycl::range local{1};
  {
  sycl::buffer<double,1> x_buf(x.data(), x.size());
  sycl::buffer<double,1> y_buf(y.data(), y.size());
  sycl::buffer<double,1> z_buf(z.data(), z.size());
  q.submit([&](sycl::handler& cgh)
  {
      auto x_d = x_buf.get_access<sycl::access::mode::read>(cgh);
      auto y_d = y_buf.get_access<sycl::access::mode::read>(cgh);
      auto z_d = z_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for(sycl::nd_range<1>{global,local}, [=](sycl::nd_item<1> i){
      		     z_d[i.get_global_linear_id()] = x_d[i.get_global_linear_id()] + y_d[i.get_global_linear_id()];
      });

  });
  }

  std::cout << "Vector addition result: \n";
  for(auto i : z) std::cout << i << " ";
  std::cout << "\n";

  //----- Task 3 -----//
  // Naive implementation with a global atomic variable
  // For more performant solutions, see: 
  // https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/kernels/reduction.html
  // Please note that clang++ does not support the built-in reduction as proposed by the Intel extensions
  std::vector<double> u(N);
  for(auto& i : x) i = 1.0;
  double acc_u = 0;
  {
  sycl::buffer<double,1> u_buf(x.data(), x.size());
  sycl::buffer<double,1> acc_buf(&acc_u, 1);

   q.submit([&](auto &h) {
		               sycl::accessor buf_acc(u_buf, h, sycl::read_only);
			       sycl::accessor sum_acc(acc_buf, h, sycl::write_only, sycl::no_init);

			             h.parallel_for(N, [=](auto index) 
				     {
					size_t glob_id = index[0];
					auto v = sycl::atomic_ref<double, 
						 sycl::memory_order::relaxed,
						 sycl::memory_scope::device,
						 sycl::access::address_space::global_space>(sum_acc[0]); 
                         v.fetch_add(buf_acc[glob_id]);
				     });
				     }
	   );
   }
  std::cout << "Reduction result:" << "\n";
  std::cout << acc_u << "\n";

  //----- Task 4 -----//
  // Consider the following article on the execution order of SYCL kernels: 
  // https://developer.codeplay.com/products/computecpp/ce/2.11.0/guides/sycl-guide/multiple-kernels
  // Under the assumption that all of the kernel functions you have written in Tasks 1 to 3 are submitted
  // to the same queue, determine which of them may be executed concurrently.
  
  return EXIT_SUCCESS;

}
