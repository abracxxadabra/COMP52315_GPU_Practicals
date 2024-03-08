#include <iostream>
#include <sycl/sycl.hpp>

int main()
{
    auto devices = sycl::device::get_devices();
    for(auto d : devices) std::cout << d.get_info<sycl::info::device::name>() << "\n";

    sycl::host_selector cpu;
    sycl::gpu_selector gpu;
    sycl::queue q(gpu);

    std::cout << "Device: "
	      << q.get_device().get_info<sycl::info::device::name>()
              << "\n";
    return EXIT_SUCCESS;
}
