#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include "simple_cuda.cuh"
#define MAXBLOCKS 65535

void simple_cuda(bool* startingGrid, bool* finalGrid, int N, int maxGen)
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

  dim3 threadNum(16, 16);
  dim3 blocks(N / threadNum.x + 1, N / threadNum.y + 1);

  cudaEvent_t startTimeDevice, endTimeDevice;
  cudaEventCreate(&startTimeDevice);
  cudaCheckErrors("Event Initialization Error");
  cudaEventCreate(&endTimeDevice);
  cudaCheckErrors("Event Initialization Error");

  cudaEventRecord(startTimeDevice, 0);
  /* Copy the initial grid to the device. */
  cudaMemcpy(currentGridDevice,startingGrid, arraySize * sizeof(bool), cudaMemcpyHostToDevice);
  for (int i = 0; i < maxGen; ++i)
  {
    // Copy the Contents of the current and the next grid
    simpleNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice, N);
    cudaCheckErrors("Exec Error");
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy(finalGrid, currentGridDevice, arraySize * sizeof(bool), cudaMemcpyDeviceToHost);

  cudaEventRecord(endTimeDevice, 0);
  cudaEventSynchronize(endTimeDevice);

  float time;
  cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);
  std::cout << "GPU Execution Time is = " << time / 1000.0f  << std::endl;

  cudaFree(currentGridDevice);
  cudaFree(nextGridDevice);
  cudaDeviceReset();

  return;
}

__global__ void simpleNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index = row * N + col;
  if (index > N * N)
    return;

  int x = index % N;
  int y = (index - x) / N;
  size_t up = ( (y + N - 1) % N) * N;
  size_t center = y * N;
  size_t down = ((y + 1) % N) * N;
  size_t left = (x + N - 1) % N;
  size_t right = (x + 1) % N;

  int livingNeighbors = calcNeighborsKernel(currentGrid, x, left, right, center, up, down);
  nextGrid[center + x] = livingNeighbors == 3 ||
    (livingNeighbors == 2 && currentGrid[x + center]) ? 1 : 0;

  return;
}

__device__ int calcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center,
    int up, int down)
{
  return currentGrid[left + up] + currentGrid[x + up]
      + currentGrid[right + up] + currentGrid[left + center]
      + currentGrid[right + center] + currentGrid[left + down]
      + currentGrid[x + down] + currentGrid[right + down];
}

