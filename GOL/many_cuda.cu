#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include "many_cuda.cuh"
#define MAXBLOCKS 512
#define CELLPERTHR 8

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

  dim3 threadNum(512);
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
  std::string prefix("[Naive Many Cells per Thread GPU]: ");
  std::cout << std::endl << prefix << "Execution Time is = <"
    << time / 1000.0f  << "> seconds" << std::endl;
  utilities::count(*finalGrid, N, N, prefix);


  // Execute the second version of the many cells per thread gpu implementation.
  threadNum = dim3(16, 8);
  blocks = dim3(std::min(N / (threadNum.x * CELLPERTHR) + 1, (unsigned int)MAXBLOCKS),
      std::min(N / (threadNum.y * CELLPERTHR) + 1, (unsigned int)MAXBLOCKS));

  cudaEventRecord(startTimeDevice, 0);
  // Copy the initial grid to the device.
  cudaMemcpy(currentGridDevice, *startingGrid, arraySize * sizeof(bool), cudaMemcpyHostToDevice);
  for (int i = 0; i < maxGen; ++i)
  {
    // Copy the Contents of the current and the next grid
    multiNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice, N);
    cudaCheckErrors("Exec Error");
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy(*finalGrid, currentGridDevice, arraySize * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaCheckErrors("Final MemCpy Error");

  cudaEventRecord(endTimeDevice, 0);
  cudaEventSynchronize(endTimeDevice);
  cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);

  prefix = std::string("[Optimized Many Cells per Thread GPU Version]: ");
  std::cout << std::endl << prefix << "Execution Time is = <"
    << time / 1000.0f  << "> seconds" << std::endl;
  utilities::count(*finalGrid, N, N, prefix);

  cudaFree(currentGridDevice);
  cudaFree(nextGridDevice);
  cudaDeviceReset();

  return;
}

__global__ void multiNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  // A 2D array contaning the data that will be used to calculate
  // the next generation for the current iteration of the game.
  bool localGrid[CELLPERTHR + 2][CELLPERTHR + 2];

  size_t row = (blockIdx.y * blockDim.y + threadIdx.y)  * CELLPERTHR;
  size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * CELLPERTHR;

  // Copy all the necessary cells for calculating the next generation
  // area for the current thread.
  // This copy operation is performed so that we may avoid repeating
  // memory read operations.
  for (int i = 0; i < CELLPERTHR + 2; i++)
  {
    int y = (row + i - 1 + N) % N * N;
    for (int j = 0; j < CELLPERTHR + 2; j++)
    {
      int x = (col + j - 1 + N) % N;
      localGrid[i][j] = currentGrid[y + x];
    }
  }

  // Apply the rules of the game of life

  for (int i = 1; i <= CELLPERTHR; i++)
  {
    int y = (row + i - 1 + N) % N;
    for (int j = 1; j <= CELLPERTHR; j++)
    {
      int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j]
        + localGrid[i - 1][j + 1] + localGrid[i][j - 1]
        + localGrid[i][j + 1] + localGrid[i + 1][j - 1] + localGrid[i + 1][j]
        + localGrid[i + 1][j + 1];
      int x = (col + j - 1 + N) % N;
      nextGrid[y * N + x] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0;
    }
  }

  return;
}

__global__ void manyNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  /* int y = blockIdx.x; */
  /* size_t up = ((y + N - 1) % N) * N; */
  /* size_t center = y * N; */
  /* size_t down = ((y + 1) % N) * N; */

  /* for (size_t cellId = threadIdx.x; */
      /* cellId < N; */
      /* cellId += blockDim.x) { */

    /* size_t left = (cellId + N - 1) % N; */
    /* size_t right = (cellId + 1) % N; */

    /* int livingNeighbors = manycalcNeighborsKernel(currentGrid, cellId, left, right, center, up, down); */
    /* nextGrid[center + cellId] = livingNeighbors == 3 || */
      /* (livingNeighbors == 2 && currentGrid[cellId + center]) ? 1 : 0; */
  /* } */
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

