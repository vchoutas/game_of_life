#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include "many_cuda.cuh"
#define MAXBLOCKS 65535
#define CELLPERTHR 50

void many_cuda(bool** startingGrid, bool** finalGrid, int N, int maxGen)
{
  const size_t arraySize = N* N;

  bool* currentGridDevice;
  bool* nextGridDevice;

  cudaMalloc((void**) &currentGridDevice, arraySize);
  cudaCheckErrors("Device memory Allocation Error!");

  cudaMalloc((void**) &nextGridDevice, arraySize);
  cudaCheckErrors("Device memory Allocation Error!");

  if (currentGridDevice == NULL || nextGridDevice == NULL)
  {
    std::cout << "Unable to allocate Device Memory!" << std::endl;
    return;
  }

  dim3 threadNum(256);
  dim3 blocks(std::min(N * N / threadNum.x + 1, (unsigned int)MAXBLOCKS));

  cudaEvent_t startTimeDevice, endTimeDevice;
  cudaEventCreate(&startTimeDevice);
  cudaCheckErrors("Event Initialization Error");
  cudaEventCreate(&endTimeDevice);
  cudaCheckErrors("Event Initialization Error");

  cudaEventRecord(startTimeDevice, 0);
  /* Copy the initial grid to the device. */
  cudaMemcpy(currentGridDevice, *startingGrid, arraySize * sizeof(bool), cudaMemcpyHostToDevice);
  for (int i = 0; i < maxGen; ++i)
  {
    // Copy the Contents of the current and the next grid
    manyNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice, N);
    cudaCheckErrors("Exec Error");
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy(*finalGrid, currentGridDevice, arraySize * sizeof(bool), cudaMemcpyDeviceToHost);

  cudaEventRecord(endTimeDevice, 0);
  cudaEventSynchronize(endTimeDevice);

  float time;
  cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);
  std::cout << std::endl << "(Many thread)GPU Execution Time is = " << time / 1000.0f  << std::endl;

  cudaFree(currentGridDevice);
  cudaFree(nextGridDevice);
  cudaDeviceReset();

  return;
}

__global__ void manyNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{

  size_t worldSize = N * N;

  for (size_t cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      cellId < worldSize;
      cellId += blockDim.x * gridDim.x) {

    size_t x = cellId % N;
    size_t yAbs = cellId - x;
    size_t xLeft = (x + N - 1) % N;
    size_t xRight = (x + 1) % N;
    size_t yAbsUp = (yAbs + worldSize - N) % worldSize;
    size_t yAbsDown = (yAbs + N) % worldSize;

    /* size_t aliveCells = currentGrid[xLeft + yAbsUp] + currentGrid[x + yAbsUp] */
      /* + currentGrid[xRight + yAbsUp] + currentGrid[xLeft + yAbs] + currentGrid[xRight + yAbs] */
      /* + currentGrid[xLeft + yAbsDown] + currentGrid[x + yAbsDown] + currentGrid[xRight + yAbsDown]; */
    size_t aliveCells = manycalcNeighborsKernel(currentGrid, x, xLeft, xRight, yAbs,
        yAbsUp,  yAbsDown);

    nextGrid[x + yAbs] =
      aliveCells == 3 || (aliveCells == 2 && currentGrid[x + yAbs]) ? 1 : 0;
  }
}


__device__ int manycalcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center,
    int up, int down)
{
  return currentGrid[left + up] + currentGrid[x + up]
      + currentGrid[right + up] + currentGrid[left + center]
      + currentGrid[right + center] + currentGrid[left + down]
      + currentGrid[x + down] + currentGrid[right + down];
}

