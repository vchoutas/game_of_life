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

__global__ void simpleNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index = row * N + col;
  if (col < N && row < N)
  {
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

__device__ int calcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center,
    int up, int down)
{
  return currentGrid[left + up] + currentGrid[x + up]
      + currentGrid[right + up] + currentGrid[left + center]
      + currentGrid[right + center] + currentGrid[left + down]
      + currentGrid[x + down] + currentGrid[right + down];
}

