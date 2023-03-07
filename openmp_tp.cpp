#include <iostream>

int main()
{
	int x = 0, y=0, z=0;
	#pragma omp parallel
	#pragma omp single
	{
		#pragma omp task depend(out: x) //A
		{ 
			x = 1;
			std::cout << "A" << "\n";
		}
		#pragma omp task depend(in: x) //B
		{ 
			y = 2 * x; 
			std::cout << "B" << "\n";
		}
		#pragma omp task depend(in: x) //C
		{ 
			z = 1 + x;
			std::cout << "C" << "\n";
		}
		#pragma omp task depend(out: x) //D
		{ 
			x = y+z;
			std::cout << "D" << "\n";
		}
	}
	std::cout << x << "\n";
}
