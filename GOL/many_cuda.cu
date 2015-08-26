#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>

#include "many_cuda.cuh"
#define MAXBLOCKS 512
#define CELLPERTHR 2

void multiCellCuda(bool* startingGrid, int N, int maxGen)
{
  std::string prefix("[Optimized Many Cells per Thread GPU Version]: ");

  bool* finalGameGrid = new bool[N * N];
  if (finalGameGrid == NULL)
  {
    std::cout << prefix << "Could not allocate memory for the final grid array!" << std::endl;
    return;
  }

  const size_t arraySize = N * N;
  bool* currentGridDevice;
  bool* nextGridDevice;

  cudaMalloc((void**) &currentGridDevice, arraySize);
  cudaCheckErrors("Device memory Allocation Error!");

  cudaMalloc((void**) &nextGridDevice, arraySize);
  cudaCheckErrors("Device memory Allocation Error!");

  if (currentGridDevice == NULL || nextGridDevice == NULL)
  {
    std::cout << prefix << "Unable to allocate Device Memory!" << std::endl;
    return;
  }

  // Execute the second version of the many cells per thread gpu implementation.
  dim3 threadNum(16, 16);
  dim3 blocks(std::min(N / (threadNum.x * CELLPERTHR) + 1, (unsigned int)MAXBLOCKS),
      std::min(N / (threadNum.y * CELLPERTHR) + 1, (unsigned int)MAXBLOCKS));

  cudaEvent_t startTimeDevice, endTimeDevice;
  cudaEventCreate(&startTimeDevice);
  cudaCheckErrors("Event Initialization Error");
  cudaEventCreate(&endTimeDevice);
  cudaCheckErrors("Event Initialization Error");


  cudaEventRecord(startTimeDevice, 0);
  // Copy the initial grid to the device.
  cudaMemcpy(currentGridDevice, startingGrid, arraySize * sizeof(bool), cudaMemcpyHostToDevice);
  for (int i = 0; i < maxGen; ++i)
  {
    // Copy the Contents of the current and the next grid
    multiNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice, N);
    cudaCheckErrors("Exec Error");
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy(finalGameGrid, currentGridDevice, arraySize * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaCheckErrors("Final MemCpy Error");

  cudaEventRecord(endTimeDevice, 0);
  cudaEventSynchronize(endTimeDevice);

  float time;
  cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);

  std::cout << std::endl << prefix << "Execution Time is = <"
    << time / 1000.0f  << "> seconds" << std::endl;
  utilities::count(finalGameGrid, N, N, prefix);

  // Free device memory.
  cudaFree(currentGridDevice);
  cudaFree(nextGridDevice);
  cudaDeviceReset();

  // Free host memory.
  delete[] finalGameGrid;

  return;
}

void multiCellCudaGhost(bool* startingGrid, int N, int maxGen)
{
  std::string prefix("[Ghost Many Cells per Thread Version]: ");

  int GhostN = N + 2;
  bool* initialGameGrid = new bool[GhostN * GhostN];
  if (initialGameGrid == NULL)
  {
    std::cout << prefix << "Could not allocate memory for the initial grid array!" << std::endl;
    return;
  }

  utilities::generate_ghost_table(startingGrid, initialGameGrid, N);

  bool* finalGameGrid = new bool[GhostN * GhostN];
  if (finalGameGrid == NULL)
  {
    std::cout << prefix << "Could not allocate memory for the final grid array!" << std::endl;
    return;
  }

  bool* currentGridDevice;
  bool* nextGridDevice;

  size_t pitchStart;
  size_t pitchDest;

  cudaMallocPitch((void**) &currentGridDevice, &pitchStart, GhostN * sizeof(bool), GhostN);
  cudaCheckErrors("Device memory Allocation Error!");

  cudaMallocPitch((void**) &nextGridDevice, &pitchDest, GhostN * sizeof(bool), GhostN);
  cudaCheckErrors("Device memory Allocation Error!");

  if (currentGridDevice == NULL || nextGridDevice == NULL)
  {
    std::cout << prefix << "Unable to allocate Device Memory!" << std::endl;
    return;
  }

  // Execute the second version of the many cells per thread gpu implementation.
  dim3 threadNum(16, 16);
  dim3 blocks(std::min(GhostN / (CELLPERTHR * threadNum.x) + 1, (unsigned int)MAXBLOCKS),
      std::min(GhostN / (CELLPERTHR * threadNum.y) + 1, (unsigned int)MAXBLOCKS));

  dim3 ghostMatThreads(16, 1);
  dim3 ghostGridRowsSize(N / ghostMatThreads.x + 1, 1);//It will not copy the corners
  dim3 ghostGridColSize(N / ghostMatThreads.x + 1, 1);//It coppies corners tooo

  cudaEvent_t startTimeDevice, endTimeDevice;
  cudaEventCreate(&startTimeDevice);
  cudaCheckErrors("Event Initialization Error");
  cudaEventCreate(&endTimeDevice);
  cudaCheckErrors("Event Initialization Error");

  cudaEventRecord(startTimeDevice, 0);
  // Copy the initial grid to the device.
  cudaMemcpy2D(currentGridDevice, pitchStart, initialGameGrid, GhostN * sizeof(bool), GhostN * sizeof(bool)
      , GhostN, cudaMemcpyHostToDevice);
  /* cudaCheckErrors("Initial MemCpy 2d"); */
  for (int i = 0; i < maxGen; ++i)
  {
    // Update the ghost elements of the Array
    utilities::updateGhostRows<<< ghostGridRowsSize, ghostMatThreads>>>(currentGridDevice, GhostN, pitchStart);
    utilities::updateGhostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice, GhostN, pitchStart);
    utilities::updateGhostCorners<<< 1, 1 >>>(currentGridDevice, GhostN, pitchStart);
    multiCellGhostGridLoop<<< blocks, threadNum >>>(currentGridDevice, nextGridDevice, N,
         pitchStart, pitchDest);
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy2D(finalGameGrid, GhostN * sizeof(bool), currentGridDevice, pitchStart, GhostN * sizeof(bool),
      GhostN, cudaMemcpyDeviceToHost);
  /* cudaCheckErrors("Final MemCpy Error"); */

  cudaEventRecord(endTimeDevice, 0);
  cudaEventSynchronize(endTimeDevice);

  float time;
  cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);

  std::cout << std::endl << prefix << "Execution Time is = <"
    << time / 1000.0f  << "> seconds" << std::endl;
  utilities::countGhost(finalGameGrid, N, N, prefix);

  // Free device memory.
  cudaFree(currentGridDevice);
  cudaFree(nextGridDevice);
  cudaDeviceReset();

  // Free host memory.
  delete[] finalGameGrid;

  return;
}

__global__ void multiNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  // A 2D array contaning the data that will be used to calculate
  // the next generation for the current iteration of the game.
  bool localGrid[CELLPERTHR + 2][CELLPERTHR + 2];

  size_t row = (blockIdx.y * blockDim.y + threadIdx.y) * CELLPERTHR;
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

__global__ void multiGhostNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  // A 2D array contaning the data that will be used to calculate
  // the next generation for the current iteration of the game.

  size_t row = (blockIdx.y * blockDim.y + threadIdx.y) * CELLPERTHR + 1;
  size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * CELLPERTHR + 1;

  bool localGrid[CELLPERTHR + 2][CELLPERTHR + 2];
  size_t yLim = CELLPERTHR > N + 1 - row ? N + 1 - row: CELLPERTHR;
  size_t xLim = CELLPERTHR > N + 1 - col ? N + 1 - col: CELLPERTHR;
  for (size_t i = 0; i < yLim + 2; i++)
  {
    size_t y = (row + i - 1) * (N + 2);
    for (size_t j = 0; j < xLim + 2; j++)
    {
      size_t x = col + j - 1;
      localGrid[i][j] = currentGrid[y + x];
    }
  }

  for (size_t i = 1; i < yLim + 1; i++)
  {
    size_t y = __umul24(row + i - 1, N + 2);
    for (size_t j = 1; j < xLim + 1; j++)
    {
      int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j]
        + localGrid[i - 1][j + 1] + localGrid[i][j - 1]
        + localGrid[i][j + 1] + localGrid[i + 1][j - 1] + localGrid[i + 1][j]
        + localGrid[i + 1][j + 1];
      size_t x = col + j - 1;
      nextGrid[y + x] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0;
    }
  }

    /* size_t yLim = CELLPERTHR + row > N + 1 ? N + 1 : CELLPERTHR + row; */
    /* size_t xLim = CELLPERTHR + col > N + 1 ? N + 1 : CELLPERTHR + col; */
    /* for (size_t i = row; i < yLim; i++) */
    /* { */
      /* size_t y = __umul24(i, N + 2); */
      /* size_t up = __umul24(i - 1, N + 2); */
      /* size_t down = __umul24(i + 1, N + 2); */
      /* for (size_t j = col; j < xLim; j++) */
      /* { */
        /* size_t left = j - 1; */
        /* size_t right = j + 1; */
        /* int livingNeighbors = manycalcNeighborsKernel(currentGrid, j, left, right, y, up, down); */
        /* nextGrid[y + j] = livingNeighbors == 3 || */
          /* (livingNeighbors == 2 && currentGrid[y + j]) ? 1 : 0; */
      /* } */
    /* } */

  return;
}

__global__ void multiCellGhostGridLoop(bool* currentGrid, bool* nextGrid, int N, size_t pitchStart,
    size_t pitchDest)
{
  int xIndex = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int yIndex = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int xStride = __umul24(blockDim.x, gridDim.x);
  int yStride = __umul24(blockDim.y, gridDim.y);
  for (int i = yIndex; i < N + 1; i += yStride)
  {
    size_t y = __umul24(i, pitchStart);
    size_t yNext = __umul24(i, pitchDest);
    size_t up = __umul24(i - 1, pitchStart);
    size_t down = __umul24(i + 1, pitchStart);
    for (int j = xIndex; j < N + 1; j += xStride)
    {
      size_t left = j - 1;
      size_t right = j + 1;

      int livingNeighbors = calcNeighborsKernel(currentGrid, j, left, right, y, up, down);
      nextGrid[yNext + j] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && currentGrid[y + j]) ? 1 : 0;
    }
  }
  return;
}


__device__ int manycalcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center,
    int up, int down)
{
  return currentGrid[left + up] + currentGrid[x + up]
    + currentGrid[right + up] + currentGrid[left + center]
    + currentGrid[right + center] + currentGrid[left + down]
    + currentGrid[x + down] + currentGrid[right + down];
}

__device__ int calcNeighborsKernel(bool* currentGrid, size_t x, size_t left, size_t right, size_t center,
    size_t up, size_t down)
{
  return currentGrid[left + up] + currentGrid[x + up]
    + currentGrid[right + up] + currentGrid[left + center]
    + currentGrid[right + center] + currentGrid[left + down]
    + currentGrid[x + down] + currentGrid[right + down];
}
