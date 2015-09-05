#ifndef SHARED_CUDA_H
#define SHARED_CUDA_H

#include "utilities.cuh"
#include <cuda.h>
#include <cuda_runtime.h>


void singleCellSharedMem(bool* startingGrid, int N, int maxGen);
void multiCellSharedMem(bool* startingGrid, int N, int maxGen);
void multiCellSharedMemPitch(bool* startingGrid, int N, int maxGen);

__global__ void singleCellSharedMemKernel(bool* currentGrid, bool* nextGrid, int N, size_t currentGridPitch,
    size_t nextGridPitch);

__global__ void multiCellSharedMemKernel(bool* currentGrid, bool* nextGrid, int N);

__global__ void multiCellSharedMemPitchKernel(bool* currentGrid, bool* nextGrid, int N, size_t currentGridPitch,
    size_t nextGridPitch);


__global__ void multiCellSharedMemKernel2(bool* currentGrid, bool* nextGrid, int N);


__device__ inline int manycalcNeighborsKernel(bool* currentGrid, int x, int left,
    int right, int center, int up , int down);

__device__ inline int sharedCalcNeighborsKernel(bool* currentGrid, size_t x, size_t left, size_t right, size_t center,
    size_t up, size_t down);

#endif  // SHARED_CUDA_H
