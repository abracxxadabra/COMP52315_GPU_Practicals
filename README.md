# COMP52315_GPU_Practicals

Please tackle the tasks yourself first, before looking at the solutions :-)

## Start an interactive session on a GPU on NCC:
Go to the [Learn Ultra page](https://blackboard.durham.ac.uk/ultra/courses/_54359_1/outline) to find out how to connect to NCC via the commandline. Once there, execute the following to access a GPU node:
`srun -c 2 --gres=gpu:1 --partition=tpg-gpu-small --pty /bin/bash`

## Compilation of CUDA code on NCC
`module load cuda`
`nvcc your_source_code.cu -o your_executable`

## Compilation of OpenMP code with CPU and GPU support
`module load nvidia-hpc`

`nvc++ -fopenmp -mp=gpu test.cpp -o test_executable`

## Environment variables to adjust the number of threads used by OpenMP
`OMP_NUM_THREADS`

`OMP_THREAD_LIMIT`

`OMP_NUM_TEAMS`

