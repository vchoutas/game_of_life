#ifndef SIMPLE_CUDA_H
#define SIMPLE_CUDA_H

#include "utilities.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline int calcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center, int up , int down);
void simpleCudaGhost(bool* startingGrid, int N, int maxGen);
__global__ void  simpleGhostNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N);

#endif // SIMPLE_CUDA_GOL_H
