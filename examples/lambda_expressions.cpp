#include <iostream>
#include <algorithm>
#include <vector>

int main()
{
	auto a_function = [](){};
	a_function();

	auto say_hi = [](){std::cout << "Hi!" << "\n";};
	say_hi();

	std::vector<int> v{1,2,3,4,5};
	int count = std::count_if(v.begin(),v.end(),[](int i){return i < 3;});
	std::cout << "#elements < 3 in 1,2,3,4,5 is: " << count << "\n";

	int a=3;
	int b=5;
	auto f = [&]()  
	{
		std::cout << a << "\n";
		std::cout << b << "\n";
		a++;
		std::cout << "local a: " << a << "\n";
	};
	f();
	std::cout << "global a: " << a << "\n";

	return EXIT_SUCCESS;
}

