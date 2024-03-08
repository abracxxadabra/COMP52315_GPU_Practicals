#include <iostream>
#include <sycl/sycl.hpp>

#define N 10

int main()
{

    sycl::gpu_selector gpu;
    sycl::host_selector cpu;
    sycl::queue q(cpu);

    std::vector<double> x(N);
    for(auto& i : x) i=1.0;


    {
    sycl::buffer<double,1> x_buf(x.data(), x.size());

    q.submit([&](sycl::handler& cgh)
    {
        auto x_d = x_buf.get_access<sycl::access::mode::read_write>(cgh);

	// High-level parallel for with a 1D iteration space with N work items
	// Global range == N
	// Local range decided by runtime
	cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
			     x_d[i] += 1;
	});

    });
    }

    for(auto i : x) std::cout << i << " ";
    std::cout << std::endl;
}
