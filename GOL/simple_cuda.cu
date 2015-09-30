#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include "simple_cuda.cuh"
#define MAXBLOCKS 65535


void simpleCudaGhost(bool* startingGrid, int N, int maxGen)
{
  std::string prefix("[Single Cell]: ");
  int GhostN = N + 2;

  bool* initialGameGrid = new bool[(GhostN) * (GhostN)];
  if (initialGameGrid == NULL)
  {
    std::cout << prefix << "Could not allocate memory for the initial grid array!" << std::endl;
    return;
  }


  bool* finalGameGrid = new bool[(GhostN) * (GhostN)];
  if (finalGameGrid == NULL)
  {
    std::cout << prefix << "Could not allocate memory for the final grid array!" << std::endl;
    return;
  }

  utilities::generate_ghost_table(startingGrid, initialGameGrid, N);
  /* utilities::print(initialGameGrid, N + 2); */
  bool* currentGridDevice;
  bool* nextGridDevice;

  cudaMalloc((void**) &currentGridDevice, GhostN *GhostN);
  cudaCheckErrors("Device memory Allocation Error!");

  cudaMalloc((void**) &nextGridDevice, GhostN *GhostN);
  cudaCheckErrors("Device memory Allocation Error!");

  if (currentGridDevice == NULL || nextGridDevice == NULL)
  {
    std::cout << prefix << "Unable to allocate Device Memory!" << std::endl;
    return;
  }

  dim3 threadNum(16, 16);
  //imperfect division creates problems(we have to use if)
  dim3 blocks((GhostN) / threadNum.x + 1, (GhostN) / threadNum.y + 1);//CREATE MACRO CALLED CEIL

  dim3 ghostMatThreads(16, 1);
  dim3 ghostGridRowsSize(N / ghostMatThreads.x + 1, 1);
  dim3 ghostGridColSize(N / ghostMatThreads.x + 1, 1);
  cudaEvent_t startTimeDevice, endTimeDevice;
  cudaEventCreate(&startTimeDevice);
  cudaCheckErrors("Event Initialization Error");
  cudaEventCreate(&endTimeDevice);
  cudaCheckErrors("Event Initialization Error");

  cudaEventRecord(startTimeDevice, 0);
  /* Copy the initial grid to the device. */
  cudaMemcpy(currentGridDevice, initialGameGrid, GhostN * GhostN , cudaMemcpyHostToDevice);

  for (int i = 0; i < maxGen; ++i)
  {
    utilities::updateGhostRows<<< ghostGridRowsSize, ghostMatThreads>>>(currentGridDevice, GhostN,GhostN );
    utilities::updateGhostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice, GhostN, GhostN);
    utilities::updateGhostCorners<<< 1, 1 >>>(currentGridDevice, GhostN, GhostN);
    simpleGhostNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice,  N);
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy(finalGameGrid, currentGridDevice, GhostN *GhostN, cudaMemcpyDeviceToHost);

  cudaEventRecord(endTimeDevice, 0);
  cudaEventSynchronize(endTimeDevice);

  float time;
  cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);
  std::cout << std::endl << prefix << "Execution Time is = <"
    << time / 1000.0f << "> seconds" << std::endl;

  utilities::countGhost(finalGameGrid, N, N, prefix);
  /* utilities::print(finalGameGrid, N + 2); */
  cudaFree(currentGridDevice);
  cudaFree(nextGridDevice);
  cudaDeviceReset();

  delete[] finalGameGrid;
  return;
}


__global__ void simpleGhostNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if ((col < N + 1) && (row < N + 1))
  {
    size_t up = (row - 1) * (N + 2);
    size_t center = row * (N + 2);
    size_t down = (row + 1) * (N + 2);
    size_t left = col - 1;
    size_t right = col + 1;

    int livingNeighbors = calcNeighborsKernel(currentGrid, col, left, right, center, up, down);
    nextGrid[center + col] = livingNeighbors == 3 ||
      (livingNeighbors == 2 && currentGrid[center + col]) ? 1 : 0;
  }
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

