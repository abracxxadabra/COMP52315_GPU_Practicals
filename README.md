# COMP52315_GPU_Practicals

Please tackle the tasks yourself first, before looking at the solutions :-)

## Start an interactive session on a GPU on NCC:

`srun -N 1 --gres=gpu:1 --partition=tpg-gpu-small --pty /bin/bash`

## Compilation of OpenMP code with CPU and GPU support
module load nvidia-hpc

nvc++ -fopenmp -mp=gpu test.cpp -o test_executable

## Environment variables to adjust the number of threads used by OpenMP
OMP_NUM_THREADS

OMP_THREAD_LIMIT

OMP_NUM_TEAMS

