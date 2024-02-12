#include <iostream>

#define N 300000
#define P 10

void vector_add_cpu_parallel_for(double* X, double* Y, double* Z)
{

	#pragma omp parallel
	{
		#pragma omp for
		for(int i = 0; i < N; i++)
		{
			Z[i] = X[i] + Y[i];
		}
	}

}

void vector_add_cpu_parallel_for_combined(double* X, double* Y, double* Z)
{
	#pragma omp parallel for
	for(int i = 0; i < N; i++)
	{
		Z[i] = X[i] + Y[i];
	}
}

void vector_add_gpu_parallel_for_combined(double* X, double* Y, double* Z)
{
	#pragma omp target parallel for map(to:X[0:N], Y[0:N]) map(tofrom:Z[0:N])
	for(int i = 0; i < N; i++)
	{
		Z[i] = X[i] + Y[i];
	}
}

void vector_add_gpu_teams_distribute_combined(double* X, double* Y, double* Z)
{
	// Caution: order matters here - teams distribute precedes map
	#pragma omp target teams distribute map(to:X[0:N], Y[0:N]) map(tofrom:Z[0:N])
	for(int i = 0; i < N; i++)
	{
		Z[i] = X[i] + Y[i];
	}
}

void vector_add_gpu_teams_parallel_combined(double* X, double* Y, double* Z)
{
	// Caution: order matters here - teams distribute precedes map
	#pragma omp target teams distribute parallel for map(to:X[0:N], Y[0:N]) map(tofrom:Z[0:N])
	for(int i = 0; i < N; i++)
	{
		Z[i] = X[i] + Y[i];
	}
}


void init(double* X, double* Y, double* Z)
{
	for(int i = 0; i < N; i++)
	{
		X[i] = 1;
		Y[i] = 1;
		Z[i] = 0;
	}
}

int main(int argc, char* argv[])
{
	double X[N], Y[N], Z[N];

	// CPU data parallelism
	init(X,Y,Z);
	vector_add_cpu_parallel_for(X,Y,Z);
	std::cout << "vector_add_cpu_parallel_for:" << "\n";
	for(int i = 0; i < P; i++) std::cout << Z[i] << " ";
	std::cout << "\n";

	init(X,Y,Z);
	vector_add_cpu_parallel_for_combined(X,Y,Z);
	std::cout << "vector_add_cpu_parallel_for_combined:" << "\n";
	for(int i = 0; i < P; i++) std::cout << Z[i] << " ";
	std::cout << "\n";

	// GPU data parallelism
	init(X,Y,Z);
	vector_add_gpu_parallel_for_combined(X,Y,Z);
	std::cout << "vector_add_gpu_parallel_for_combined:" << "\n";
	for(int i = 0; i < P; i++) std::cout << Z[i] << " ";
	std::cout << "\n";

	init(X,Y,Z);
	vector_add_gpu_teams_distribute_combined(X,Y,Z);
	std::cout << "vector_add_gpu_teams_distribute_combined:" << "\n";
	for(int i = 0; i < P; i++) std::cout << Z[i] << " ";
	std::cout << "\n";

	init(X,Y,Z);
	vector_add_gpu_teams_parallel_combined(X,Y,Z);
	std::cout << "vector_add_gpu_teams_parallel_combined:" << "\n";
	for(int i = 0; i < P; i++) std::cout << Z[i] << " ";
	std::cout << "\n";
}
