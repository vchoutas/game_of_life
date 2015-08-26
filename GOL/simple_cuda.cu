#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include "simple_cuda.cuh"
#define MAXBLOCKS 65535

void simpleCuda(bool* startingGrid, int N, int maxGen)
{
  std::string prefix("[Naive Single Cell per Thread]: ");

  // The host array that will contain the game of life grid after maxGen generations.
  bool* finalGameGrid = new bool[N * N];
  if (finalGameGrid == NULL)
  {
    std::cout << prefix << "Could not allocate memory for the next generation grid array!" << std::endl;
    return;
  }

  // Copy the input data to
  const size_t arraySize = N* N;

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

  dim3 threadNum(16, 16);
  dim3 blocks(N / threadNum.x + 1, N / threadNum.y + 1);

  cudaEvent_t startTimeDevice, endTimeDevice;
  cudaEventCreate(&startTimeDevice);
  cudaCheckErrors("Event Initialization Error");
  cudaEventCreate(&endTimeDevice);
  cudaCheckErrors("Event Initialization Error");

  cudaEventRecord(startTimeDevice, 0);
  /* Copy the initial grid to the device. */
  cudaMemcpy(currentGridDevice, startingGrid, arraySize * sizeof(bool), cudaMemcpyHostToDevice);
  for (int i = 0; i < maxGen; ++i)
  {
    // Copy the Contents of the current and the next grid
    simpleNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice, N);
    cudaCheckErrors("Exec Error");
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy(finalGameGrid, currentGridDevice, arraySize * sizeof(bool), cudaMemcpyDeviceToHost);

  cudaEventRecord(endTimeDevice, 0);
  cudaEventSynchronize(endTimeDevice);

  float time;
  cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);
  std::cout << std::endl << prefix << "Execution Time is = <"
    << time / 1000.0f << "> seconds" << std::endl;
  utilities::count(finalGameGrid, N, N, prefix);

  cudaFree(currentGridDevice);
  cudaFree(nextGridDevice);
  cudaDeviceReset();

  delete[] finalGameGrid;

  return;
}

void simpleCudaPitch(bool* startingGrid, int N, int maxGen)
{
  std::string prefix("[Naive Single Cell per Thread Pitch]: ");

  bool* finalGameGrid = new bool[N * N];
  if (finalGameGrid == NULL)
  {
    std::cout << prefix << "Could not allocate memory for the final grid array!" << std::endl;
    return;
  }

  bool* currentGridDevice;
  bool* nextGridDevice;

  size_t pitchStart;
  size_t pitchDest;

  cudaMallocPitch((void**) &currentGridDevice, &pitchStart, N * sizeof(bool), N);
  cudaCheckErrors("Device memory Allocation Error!");

  cudaMallocPitch((void**) &nextGridDevice, &pitchDest, N * sizeof(bool), N);
  cudaCheckErrors("Device memory Allocation Error!");

  if (currentGridDevice == NULL || nextGridDevice == NULL)
  {
    std::cout << prefix << "Unable to allocate Device Memory!" << std::endl;
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
  cudaMemcpy2D(currentGridDevice, pitchStart, startingGrid, N * sizeof(bool), N * sizeof(bool)
      , N, cudaMemcpyHostToDevice);
  cudaCheckErrors("Initial Memcpy 2D Error");
  for (int i = 0; i < maxGen; ++i)
  {
    // Copy the Contents of the current and the next grid
    simpleNextGenerationKernelPitch<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice, N, pitchStart,
        pitchDest);
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy2D(finalGameGrid, N * sizeof(bool), currentGridDevice, pitchStart, N * sizeof(bool),
      N, cudaMemcpyDeviceToHost);
  cudaCheckErrors("Final Memcpy 2D Error");

  cudaEventRecord(endTimeDevice, 0);
  cudaEventSynchronize(endTimeDevice);

  float time;
  cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);
  std::cout << std::endl << prefix << "Execution Time is = <"
    << time / 1000.0f << "> seconds" << std::endl;
  utilities::count(finalGameGrid, N, N, prefix);

  cudaFree(currentGridDevice);
  cudaFree(nextGridDevice);
  cudaDeviceReset();

  delete[] finalGameGrid;
  return;
}



void simpleCudaGhostPitch(bool* startingGrid, int N, int maxGen)
{
  std::string prefix("[Ghosts Single Cell per Thread Pitch]: ");
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

  size_t pitchStart;
  size_t pitchDest;

  /* cudaMallocPitch((void**) &currentGridDevice, &pitchStart, (GhostN) * sizeof(bool), (GhostN)); */
  /* cudaCheckErrors("Device memory Allocation Error!"); */

  /* cudaMallocPitch((void**) &nextGridDevice, &pitchDest, (GhostN) * sizeof(bool), (GhostN)); */
  /* cudaCheckErrors("Device memory Allocation Error!"); */
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
  dim3 ghostGridRowsSize(N / ghostMatThreads.x + 1, 1);//It will not copy the corners
  dim3 ghostGridColSize(N / ghostMatThreads.x + 1, 1);//It coppies corners tooo

  cudaEvent_t startTimeDevice, endTimeDevice;
  cudaEventCreate(&startTimeDevice);
  cudaCheckErrors("Event Initialization Error");
  cudaEventCreate(&endTimeDevice);
  cudaCheckErrors("Event Initialization Error");

  cudaEventRecord(startTimeDevice, 0);
  /* Copy the initial grid to the device. */
  /* cudaMemcpy2D(currentGridDevice, pitchStart, initialGameGrid, GhostN * sizeof(bool), GhostN * sizeof(bool) */
      /* , GhostN, cudaMemcpyHostToDevice); */
  /* cudaCheckErrors("Initial Memcpy 2D Error"); */
  cudaMemcpy(currentGridDevice, initialGameGrid, GhostN * GhostN *sizeof(bool), cudaMemcpyHostToDevice);

  for (int i = 0; i < maxGen; ++i)
  {
    utilities::ghostRows<<< ghostGridRowsSize, ghostMatThreads>>>(currentGridDevice, GhostN);
    utilities::ghostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice, GhostN);
    utilities::ghostCorners<<< 1, 1 >>>(currentGridDevice, GhostN);
    /* simpleGhostNextGenerationKernelPitch<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice, */
        /* N, pitchStart, pitchDest); */
    simpleGhostNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice,  N);
    /* cudaDeviceSynchronize(); */
    /* cudaCheckErrors("Exec Error"); */
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  /* cudaMemcpy2D(finalGameGrid, GhostN * sizeof(bool), currentGridDevice, pitchStart, GhostN * sizeof(bool), */
      /* GhostN, cudaMemcpyDeviceToHost); */
  /* cudaCheckErrors("Final Memcpy 2D Error"); */
  cudaMemcpy(finalGameGrid, currentGridDevice, GhostN *GhostN * sizeof(bool), cudaMemcpyDeviceToHost);


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

__global__ void simpleNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index = row * N + col;
  if (col < N && row < N)
  {
    int x = index % N;
    int y = (index - x) / N;
    size_t up = ((y + N - 1) % N) * N;
    size_t center = y * N;
    size_t down = ((y + 1) % N) * N;
    size_t left = (x + N - 1) % N;
    size_t right = (x + 1) % N;

    int livingNeighbors = calcNeighborsKernel(currentGrid, x, left, right, center, up, down);
    nextGrid[center + x] = livingNeighbors == 3 ||
      (livingNeighbors == 2 && currentGrid[x + center]) ? 1 : 0;
  }
  return;
}

__global__ void simpleNextGenerationKernelPitch(bool* currentGrid, bool* nextGrid, int N,
    size_t currentGridPitch, size_t nextGridPitch)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < N && row < N)
  {
    bool* currentRow = (bool*)(currentGrid + row * currentGridPitch);

    // The row above the current one.
    size_t up = (row + N - 1) % N;
    bool* previousRow = (bool*)(currentGrid + up * currentGridPitch);
    // The row below the current one.
    size_t down = (row + 1) % N;
    bool* nextRow = (bool*)(currentGrid + down * currentGridPitch);
    // Get the index for the left column
    size_t left = (col + N - 1) % N;
    // Get the index of the right column
    size_t right = (col + 1) % N;


    int livingNeighbors = previousRow[left] + previousRow[col] + previousRow[right]
      + currentRow[left] + currentRow[right] + nextRow[left] + nextRow[col] + nextRow[right];

    bool* nextGridRow = (bool*)(nextGrid + row * nextGridPitch);
    nextGridRow[col] = livingNeighbors == 3 ||
      (livingNeighbors == 2 && currentRow[col]) ? 1 : 0;
  }
  return;
}

__global__ void ghostRows(bool* Grid, int N)//Does not  copy corners twp
{
  int x = blockDim.x * blockIdx.x + threadIdx.x + 1;
  if (x < N - 1)
  {
    //The first and last columns are to be wrriten
    Grid[toLinearIndex(N - 1, x, N)] = Grid[toLinearIndex(1, x, N)];  //write bottom to top
    Grid[toLinearIndex(0, x, N)] = Grid[toLinearIndex(N - 2, x, N)];  //write top to bottom
  }
}
__global__ void ghostCols(bool* Grid,int N)//Does not copy corners
{
  int y = blockDim.x * blockIdx.x + threadIdx.x + 1;
  if (y< N-1)
  {
    //std:cout<<id;
    Grid[toLinearIndex(y, N - 1, N)] = Grid[toLinearIndex(y, 1, N)];  //write left  to   right
    Grid[toLinearIndex(y, 0, N)] = Grid[toLinearIndex(y, N - 2, N)];  //write right  to left

  }
}

__global__ void ghostCorners(bool* grid, int N)
{
  grid[toLinearIndex(0, 0, N)] = grid[toLinearIndex(N-2, N - 2, N)];//(0,0)-->(N-2,N-2)
  grid[toLinearIndex(N-1, N - 1, N)] = grid[toLinearIndex(1, 1, N)];//(N-1,N-1)-->(1,1)
  grid[toLinearIndex(0, N - 1, N)] = grid[toLinearIndex(N - 2, 1, N)];//(0,N-1)-->(N-2,1)
  grid[toLinearIndex(N - 1, 0, N)] = grid[toLinearIndex(1, N - 2, N)];//(N-1,0)-->(1,N-2)
}


__global__ void simpleGhostNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  /* int index = row * (N + 2) + col; */
  if ((col < N + 1) && (row < N + 1))
  {
    int x = col;
    int y = row;
    size_t up = ((y - 1)) * (N+2);
    size_t center = y * (N+2);
    size_t down = ((y + 1) ) * (N+2);
    size_t left = (x - 1);
    size_t right = (x + 1);

    int livingNeighbors = calcNeighborsKernel(currentGrid, x, left, right, center, up, down);
    nextGrid[center + x] = livingNeighbors == 3 ||
      (livingNeighbors == 2 && currentGrid[x + center]) ? 1 : 0;
  }
  return;
}

__global__ void simpleGhostNextGenerationKernelPitch(bool* currentGrid, bool* nextGrid, int N,
    size_t currentGridPitch, size_t nextGridPitch)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if ((col < N + 1) && (row < N + 1))
  {
    bool* currentRow = (bool*)(currentGrid + row * currentGridPitch);

    /* The row above the current one. */
    size_t up = (row - 1);
    bool* previousRow = (bool*)(currentGrid + up * currentGridPitch);
    /* The row below the current one. */
    size_t down = (row + 1);
    bool* nextRow = (bool*)(currentGrid + down * currentGridPitch);
    /* Get the index for the left column */
    size_t left = (col - 1);
    /* Get the index of the right column */
    size_t right = (col + 1) ;


    int livingNeighbors = previousRow[left] + previousRow[col] + previousRow[right]
      + currentRow[left] + currentRow[right] + nextRow[left] + nextRow[col] + nextRow[right];

    bool* nextGridRow = (bool*)(nextGrid + row * nextGridPitch);
    nextGridRow[col] = livingNeighbors == 3 ||
      (livingNeighbors == 2 && currentRow[col]) ? 1 : 0;
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

