# Parallel Implementation of Strassen Matrix Multiplication algorithm using CUDA


The algorithm for Strassen Matrix Multiplication can be found here: https://en.wikipedia.org/wiki/Strassen_algorithm

**Platform:** CUDA

**Compile command:** nvcc -arch=sm_80 -ccbin=icc -std=c++17 -x cu -lcublas -o strassen_matmul.exe strassen_matmul.cpp

**Execution:** ./strassen_matmul.exe k k’ blockdim2d

where k -> matrix dimension, k’ -> terminal matrix dimension, blockdim2d -> dim of 2d block in terms of number of threads
Example: blockdim2d = 32 => #threads = 32 x 32 = 1024
blockdim2d range : [1, 32]
