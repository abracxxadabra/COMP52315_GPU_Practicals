#include<iostream>
#include<string>

void vector_add_seq(const float* x, const float* y, float* z, const int N)
{

	for(int i = 0; i < N; i++)
	{
		z[i] = x[i] + y[i];
	}

}

__global__ void vector_add_simple_gpu(const float* x, const float* y, float* z, const int N)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread_id < N)
	{
		z[thread_id] = x[thread_id] + y[thread_id];
	}
}

__global__ void vector_add_grid_strided_gpu(const float* x, const float* y, float* z, const int N)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_stride = blockDim.x * gridDim.x;

	for(int i = thread_id; i < N; i+=grid_stride)
	{
		z[i] = x[i] + y[i];
	}
}

int main(int argc, char* argv[])
{
	int N = std::stoi(argv[1]);

	float* x = new float[N];
	float* y = new float[N];
	float* z = new float[N];

	for(int i =0; i<N; i++)
	{
		x[i] = 1.0;
		y[i] = 1.0;
	}

	std::cout << "Sequential vector add:\n";
	vector_add_seq(x, y, z, N);
	for(int i = 0; i < N; i++)
	{
		std::cout << z[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "Simple GPU vector add:\n";

	float* x_d;
	float* y_d;
	float* z_d;

	for(int i =0; i<N; i++)
	{
		x[i] = 2.0;
		y[i] = 2.0;
	}

	cudaMalloc(&x_d, sizeof(float) * N);
	cudaMalloc(&y_d, sizeof(float) * N);
	cudaMalloc(&z_d, sizeof(float) * N);

	cudaMemcpy(x_d, x, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(float) * N, cudaMemcpyHostToDevice);

	int block_size = 64;
	int grid_size = N/block_size + (N%block_size != 0);

	vector_add_simple_gpu<<<grid_size, block_size>>>(x_d, y_d, z_d, N);
	cudaDeviceSynchronize();

	cudaMemcpy(z, z_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
  	
	for(int i = 0; i < N; i++)
	{
		std::cout << z[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "Grid-strided loop GPU vector add:\n";
	block_size = std::stoi(argv[2]);
	grid_size = std::stoi(argv[2]);

	for(int i =0; i<N; i++)
	{
		x[i] = 3.0;
		y[i] = 3.0;
	}
	cudaMemcpy(x_d, x, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(float) * N, cudaMemcpyHostToDevice);

	vector_add_grid_strided_gpu<<<grid_size, block_size>>>(x_d, y_d, z_d, N);
	cudaDeviceSynchronize();

	cudaMemcpy(z, z_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++)
	{
		std::cout << z[i] << " ";
	}
	std::cout << std::endl;
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);

	delete[] x;
	delete[] y;
	delete[] z;
	return 0;
}
