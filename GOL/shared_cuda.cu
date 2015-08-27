#include <iostream>

#include "shared_cuda.cuh"
#include "utilities.cuh"

#define TILE_SIZE_X 20
#define TILE_SIZE_Y 16
#define CELLS_PER_THR 2
#define MAXBLOCKS 512


void singleCellSharedMem(bool* startingGrid, int N, int maxGen)
{
  std::string prefix("[Shared Memory Single Cell per Thread]: ");

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
  dim3 threadNum(TILE_SIZE_X, TILE_SIZE_Y);
  dim3 blocks(std::min(N / (threadNum.x) + 1, (unsigned int)MAXBLOCKS),
      std::min(N / (threadNum.y) + 1, (unsigned int)MAXBLOCKS));

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
    singleCellSharedMemKernel<<< blocks, threadNum >>>(currentGridDevice, nextGridDevice, N,
        pitchStart, pitchDest);
    /* cudaDeviceSynchronize(); */
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

__global__ void singleCellSharedMemKernel(bool* currentGrid, bool* nextGrid, int N, size_t pitchStart,
    size_t pitchDest)
{
  size_t row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = threadIdx.y + 1;
  int j = threadIdx.x + 1;

  __shared__ bool localGrid[TILE_SIZE_Y + 2][TILE_SIZE_X + 2];


  localGrid[i][j] = currentGrid[row * pitchStart + col];
  if (i == 1)
    localGrid[i - 1][j] = currentGrid[(row - 1) * pitchStart + col];
  if (j == 1)
    localGrid[i][j - 1] = currentGrid[row * pitchStart + col - 1];
  if (i == TILE_SIZE_Y)
    localGrid[i + 1][j] = currentGrid[(row + 1) * pitchStart + col];
  if (j == TILE_SIZE_X)
    localGrid[i][j + 1] = currentGrid[row * pitchStart + col + 1];

  if (i == 1 && j == 1)
    localGrid[i - 1][j - 1] = currentGrid[(row - 1) * pitchStart + col - 1];
  if (i == 1 && j == TILE_SIZE_X)
    localGrid[i - 1][j + 1] = currentGrid[(row - 1) * pitchStart + col + 1];
  if (i == TILE_SIZE_Y && j == 1)
    localGrid[i + 1][j - 1] = currentGrid[(row + 1) * pitchStart + col - 1];
  if (i == TILE_SIZE_Y && j == TILE_SIZE_X)
    localGrid[i + 1][j + 1] = currentGrid[(row + 1) * pitchStart + col + 1];

  /* __syncthreads(); */

  /* localGrid[i][j] = currentGrid[row * pitchStart + col]; */
  __syncthreads();

  /* if (i > 0 && i <= TILE_SIZE && j > 0 && j <= TILE_SIZE) */
  /* { */
    int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j]
      + localGrid[i - 1][j + 1] + localGrid[i][j - 1] + localGrid[i][j + 1]
      + localGrid[i + 1][j - 1] + localGrid[i + 1][j] + localGrid[i + 1][j + 1];
    nextGrid[row * pitchDest + col] = livingNeighbors == 3 ||
      (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0;
  /* } */
  return;
  /* int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j] */
    /* + localGrid[i - 1][j + 1] + localGrid[i][j - 1] + localGrid[i][j + 1] */
    /* + localGrid[i + 1][j - 1] + localGrid[i + 1][j] + localGrid[i + 1][j + 1]; */
  /* nextGrid[row * pitchDest + col] = livingNeighbors == 3 || */
    /* (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0; */

  /* int xIndex = blockIdx.x * blockDim.x + threadIdx.x + 1; */
  /* int yIndex = blockIdx.y * blockDim.y + threadIdx.y + 1; */
  /* int xStride = __umul24(blockDim.x, gridDim.x); */
  /* int yStride = __umul24(blockDim.y, gridDim.y); */
  /* for (int i = blockIdx.y * blockIdx.y + threadIdx.y; */
      /* i < N + 1; i += blockDim.y * gridDim.y) */
  /* { */
    /* size_t y = __umul24(i, pitchStart); */
    /* size_t yNext = __umul24(i, pitchDest); */
    /* size_t up = __umul24(i - 1, pitchStart); */
    /* size_t down = __umul24(i + 1, pitchStart); */
    /* for (int j = xIndex; j < N + 1; j += xStride) */
    /* { */
      /* size_t left = j - 1; */
      /* size_t right = j + 1; */

      /* int livingNeighbors = sharedCalcNeighborsKernel(currentGrid, j, left, right, y, up, down); */
      /* nextGrid[yNext + j] = livingNeighbors == 3 || */
        /* (livingNeighbors == 2 && currentGrid[y + j]) ? 1 : 0; */
    /* } */
  /* } */

}

__device__ int sharedCalcNeighborsKernel(bool* currentGrid, size_t x, size_t left, size_t right, size_t center,
    size_t up, size_t down)
{
  return currentGrid[left + up] + currentGrid[x + up]
    + currentGrid[right + up] + currentGrid[left + center]
    + currentGrid[right + center] + currentGrid[left + down]
    + currentGrid[x + down] + currentGrid[right + down];
}
