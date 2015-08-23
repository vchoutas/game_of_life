#ifndef MANY_CUDA_H
#define MANY_CUDA_H

#include "utilities.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void manyNextGenerationKernel(bool* currentGrid, bool* nextGrid, const int N);
__device__ inline int manycalcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center, int up , int down);
void many_cuda(bool** startingGrid, bool** finalGrid, int N, int maxGen);

#endif // MANY_CUDA_GOL_H
