#ifndef SHARED_CUDA_H
#define SHARED_CUDA_H

#include "utilities.cuh"
#include <cuda.h>
#include <cuda_runtime.h>


void singleCellSharedMem(bool* startingGrid, int N, int maxGen);
void multiCellSharedMem(bool* startingGrid, int N, int maxGen);

__global__ void singleCellSharedMemKernel(bool* currentGrid, bool* nextGrid, int N, size_t currentGridPitch,
    size_t nextGridPitch);

__global__ void multiCellSharedMemKernel(bool* currentGrid, bool* nextGrid, int N);

#endif  // SHARED_CUDA_H
