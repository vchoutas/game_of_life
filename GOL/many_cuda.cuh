#ifndef MANY_CUDA_H
#define MANY_CUDA_H

#include "utilities.h"
#include <cuda.h>
#include <cuda_runtime.h>

// Device Functions.
__global__ void manyNextGenerationKernel(bool* currentGrid, bool* nextGrid, const int N);
__global__ void multiNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N);
__device__ inline int manycalcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center, int up , int down);

// Host Functions.
void multiCellCudaNaive(bool* startingGrid, int N, int maxGen);
void multiCellCuda(bool* startingGrid, int N, int maxGen);

#endif // MANY_CUDA_GOL_H
