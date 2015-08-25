#ifndef SIMPLE_CUDA_H
#define SIMPLE_CUDA_H

#include "utilities.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void simpleNextGenerationKernel(bool* currentGrid, bool* nextGrid, const int N);
__device__ inline int calcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center, int up , int down);
void simpleCuda(bool* startingGrid, int N, int maxGen);
void simpleCudaPitch(bool* startingGrid, int N, int maxGen);
__global__ void simpleNextGenerationKernelPitch(bool* currentGrid, bool* nextGrid, int N,
    size_t currentGridPitch, size_t nextGridPitch);

#endif // SIMPLE_CUDA_GOL_H
