#ifndef MANY_CUDA_H
#define MANY_CUDA_H

#include "utilities.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

// Device Functions.
__global__ void multiCellGhostGridLoop(bool* currentGrid, bool* nextGrid, int N, size_t pitchStart,
    size_t pitchDest);
__global__ void multiCellKernel(bool* currentGrid, bool* nextGrid, int N);

__device__ int calcNeighborsKernel(bool* currentGrid, size_t x, size_t left, size_t right, size_t center,
    size_t up, size_t down);

// Host Functions.
void gridLoopGhost(bool* startingGrid, int N, int maxGen);
void multiCellCudaGhost(bool* startingGrid, int N, int maxGen);
#endif // MANY_CUDA_GOL_H
