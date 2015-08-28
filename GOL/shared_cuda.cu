#include <iostream>

#include "shared_cuda.cuh"
#include "utilities.cuh"

#define TILE_SIZE 16
#define TILE_SIZE_X 16
#define TILE_SIZE_Y 16
#define CELLS_PER_THR 2
#define MAXBLOCKS 512


void singleCellSharedMem(bool* startingGrid, int N, int maxGen){
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

  size_t currentGridPitch;
  size_t nextGridPitch;

  cudaMallocPitch((void**) &currentGridDevice, &currentGridPitch, GhostN * sizeof(bool), GhostN);
  cudaCheckErrors("Device memory Allocation Error!");

  cudaMallocPitch((void**) &nextGridDevice, &nextGridPitch, GhostN * sizeof(bool), GhostN);
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
  cudaMemcpy2D(currentGridDevice, currentGridPitch, initialGameGrid, GhostN * sizeof(bool), GhostN * sizeof(bool)
      , GhostN, cudaMemcpyHostToDevice);
  /* cudaCheckErrors("Initial MemCpy 2d"); */
  for (int i = 0; i < maxGen; ++i)
  {
    // Update the ghost elements of the Array
    utilities::updateGhostRows<<< ghostGridRowsSize, ghostMatThreads>>>(currentGridDevice, GhostN, currentGridPitch);
    utilities::updateGhostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice, GhostN, currentGridPitch);
    utilities::updateGhostCorners<<< 1, 1 >>>(currentGridDevice, GhostN, currentGridPitch);
    singleCellSharedMemKernel<<< blocks, threadNum >>>(currentGridDevice, nextGridDevice, N,
        currentGridPitch, nextGridPitch);
    /* cudaDeviceSynchronize(); */
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy2D(finalGameGrid, GhostN * sizeof(bool), currentGridDevice, currentGridPitch, GhostN * sizeof(bool),
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


void multiCellSharedMemPitch(bool* startingGrid, int N, int maxGen){
  std::string prefix("[Shared Memory Multiple Cells per Thread Pitch]: ");

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

  size_t currentGridPitch;
  size_t nextGridPitch;

  cudaMallocPitch((void**) &currentGridDevice, &currentGridPitch, GhostN * sizeof(bool), GhostN);
  cudaCheckErrors("Device memory Allocation Error!");

  cudaMallocPitch((void**) &nextGridDevice, &nextGridPitch, GhostN * sizeof(bool), GhostN);
  cudaCheckErrors("Device memory Allocation Error!");

  if (currentGridDevice == NULL || nextGridDevice == NULL)
  {
    std::cout << prefix << "Unable to allocate Device Memory!" << std::endl;
    return;
  }

  // Execute the second version of the many cells per thread gpu implementation.
  dim3 threadNum(TILE_SIZE_X, TILE_SIZE_Y);
  dim3 blocks(std::min(GhostN / (threadNum.x * CELLS_PER_THR) + 1, (unsigned int)MAXBLOCKS),
      std::min(GhostN / (threadNum.y * CELLS_PER_THR) + 1, (unsigned int)MAXBLOCKS));

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
  cudaMemcpy2D(currentGridDevice, currentGridPitch, initialGameGrid, GhostN * sizeof(bool), GhostN * sizeof(bool)
      , GhostN, cudaMemcpyHostToDevice);
  /* cudaCheckErrors("Initial MemCpy 2d"); */
  for (int i = 0; i < maxGen; ++i)
  {
    // Update the ghost elements of the Array
    utilities::updateGhostRows<<< ghostGridRowsSize, ghostMatThreads>>>(currentGridDevice, GhostN, currentGridPitch);
    utilities::updateGhostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice, GhostN, currentGridPitch);
    utilities::updateGhostCorners<<< 1, 1 >>>(currentGridDevice, GhostN, currentGridPitch);
    multiCellSharedMemPitchKernel<<< blocks, threadNum >>>(currentGridDevice, nextGridDevice, N,
        currentGridPitch, nextGridPitch);
    cudaDeviceSynchronize();
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy2D(finalGameGrid, GhostN * sizeof(bool), currentGridDevice, currentGridPitch, GhostN * sizeof(bool),
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

void multiCellSharedMem(bool* startingGrid, int N, int maxGen){
  std::string prefix("[Shared Memory Multiple Cells per Thread ]: ");
  int GhostN = N + 2;

  bool* initialGameGrid = new bool[(GhostN) * (GhostN)];
  if (initialGameGrid == NULL){
    std::cout << prefix << "Could not allocate memory for the initial grid array!" << std::endl;
    return;
  }

  bool* finalGameGrid = new bool[(GhostN) * (GhostN)];
  if (finalGameGrid == NULL){
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

  if (currentGridDevice == NULL || nextGridDevice == NULL){
    std::cout << prefix << "Unable to allocate Device Memory!" << std::endl;
    return;
  }

  dim3 threadNum(TILE_SIZE, TILE_SIZE);
  //imperfect division creates problems(we have to use if)
  dim3 blocks(GhostN/(threadNum.x * CELLS_PER_THR)+ 1, GhostN/( threadNum.y * CELLS_PER_THR) + 1);//CREATE MACRO CALLED CEIL

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
  cudaMemcpy(currentGridDevice, initialGameGrid, GhostN * GhostN, cudaMemcpyHostToDevice);

  for (int i = 0; i < maxGen; ++i)
  {
    utilities::updateGhostRows<<< ghostGridRowsSize, ghostMatThreads>>>(currentGridDevice, GhostN, GhostN);
    utilities::updateGhostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice, GhostN, GhostN );
    utilities::updateGhostCorners<<< 1, 1 >>>(currentGridDevice, GhostN, GhostN );
    multiCellSharedMemKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice,  N);
    SWAP(currentGridDevice, nextGridDevice);
  }
  // Copy the final grid back to the host memory.
  cudaMemcpy(finalGameGrid, currentGridDevice, GhostN *GhostN , cudaMemcpyDeviceToHost);

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

__global__ void singleCellSharedMemKernel(bool* currentGrid, bool* nextGrid, int N, size_t currentGridPitch,
    size_t nextGridPitch){
  size_t row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = threadIdx.y + 1;
  int j = threadIdx.x + 1;

  __shared__ bool localGrid[TILE_SIZE_Y + 2][TILE_SIZE_X + 2];


  localGrid[i][j] = currentGrid[row * currentGridPitch + col];
  if (i == 1)
    localGrid[i - 1][j] = currentGrid[(row - 1) * currentGridPitch + col];
  if (j == 1)
    localGrid[i][j - 1] = currentGrid[row * currentGridPitch + col - 1];
  if (i == TILE_SIZE_Y)
    localGrid[i + 1][j] = currentGrid[(row + 1) * currentGridPitch + col];
  if (j == TILE_SIZE_X)
    localGrid[i][j + 1] = currentGrid[row * currentGridPitch + col + 1];

  if (i == 1 && j == 1)
    localGrid[i - 1][j - 1] = currentGrid[(row - 1) * currentGridPitch + col - 1];
  if (i == 1 && j == TILE_SIZE_X)
    localGrid[i - 1][j + 1] = currentGrid[(row - 1) * currentGridPitch + col + 1];
  if (i == TILE_SIZE_Y && j == 1)
    localGrid[i + 1][j - 1] = currentGrid[(row + 1) * currentGridPitch + col - 1];
  if (i == TILE_SIZE_Y && j == TILE_SIZE_X)
    localGrid[i + 1][j + 1] = currentGrid[(row + 1) * currentGridPitch + col + 1];

  /* __syncthreads(); */

  /* localGrid[i][j] = currentGrid[row * currentGridPitch + col]; */
  __syncthreads();

  /* if (i > 0 && i <= TILE_SIZE && j > 0 && j <= TILE_SIZE) */
  /* { */
    int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j]
      + localGrid[i - 1][j + 1] + localGrid[i][j - 1] + localGrid[i][j + 1]
      + localGrid[i + 1][j - 1] + localGrid[i + 1][j] + localGrid[i + 1][j + 1];
    nextGrid[row * nextGridPitch + col] = livingNeighbors == 3 ||
      (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0;
  /* } */
  return;
  /* int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j] */
    /* + localGrid[i - 1][j + 1] + localGrid[i][j - 1] + localGrid[i][j + 1] */
    /* + localGrid[i + 1][j - 1] + localGrid[i + 1][j] + localGrid[i + 1][j + 1]; */
  /* nextGrid[row * nextGridPitch + col] = livingNeighbors == 3 || */
    /* (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0; */

  /* int xIndex = blockIdx.x * blockDim.x + threadIdx.x + 1; */
  /* int yIndex = blockIdx.y * blockDim.y + threadIdx.y + 1; */
  /* int xStride = __umul24(blockDim.x, gridDim.x); */
  /* int yStride = __umul24(blockDim.y, gridDim.y); */
  /* for (int i = blockIdx.y * blockIdx.y + threadIdx.y; */
      /* i < N + 1; i += blockDim.y * gridDim.y) */
  /* { */
    /* size_t y = __umul24(i, currentGridPitch); */
    /* size_t yNext = __umul24(i, nextGridPitch); */
    /* size_t up = __umul24(i - 1, currentGridPitch); */
    /* size_t down = __umul24(i + 1, currentGridPitch); */
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

__global__ void multiCellSharedMemKernel(bool* currentGrid, bool* nextGrid, int N){

  size_t g_row = (blockIdx.y * blockDim.y + threadIdx.y) * CELLS_PER_THR ;
  size_t g_col = (blockIdx.x * blockDim.x + threadIdx.x) * CELLS_PER_THR ;

  size_t l_row = threadIdx.y * CELLS_PER_THR ;
  size_t l_col = threadIdx.x * CELLS_PER_THR;


  __shared__ bool localGrid[TILE_SIZE*CELLS_PER_THR + 2][TILE_SIZE*CELLS_PER_THR + 2];
  for (size_t i = 0; i < CELLS_PER_THR ; i++)//Must change
  {
    size_t y = (g_row + i ) * (N + 2);
    for (size_t j = 0; j < CELLS_PER_THR ; j++)//Must change
    {
      size_t x = g_col + j ;
      localGrid[j + l_row][i +l_col ] = currentGrid[y + x];
    }
  }
  __syncthreads();


for (size_t i = 1; i < CELLS_PER_THR + 1; i++)
  {
    size_t y = __umul24(g_row + i , N + 2);
    size_t li = i + l_col;
    for (size_t j = 1; j < CELLS_PER_THR + 1; j++)
    {

      size_t lj = j+ l_row;
      int livingNeighbors = localGrid[li - 1][lj - 1] + localGrid[li - 1][lj]
        + localGrid[li - 1][lj + 1] + localGrid[li][lj - 1]
        + localGrid[li][lj + 1] + localGrid[li + 1][lj - 1] + localGrid[li + 1][lj]
        + localGrid[li + 1][lj + 1];


      size_t x = g_col + j ;
      nextGrid[y + x] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && localGrid[li][lj]) ? 1 : 0;
    }
  }

  return;
}


__global__ void multiCellSharedMemPitchKernel(bool* currentGrid, bool* nextGrid, int N, size_t currentGridPitch,
    size_t nextGridPitch){

  //Copy the neccesary cells to the shared Memory
  int xIndex = blockIdx.x * blockDim.x + threadIdx.x +1;
  int yIndex = blockIdx.y * blockDim.y + threadIdx.y +1;

  int xStride = __umul24(blockDim.x, gridDim.x);
  int yStride = __umul24(blockDim.y, gridDim.y);

  /* int xStride = blockDim.x * gridDim.x; */
  /* int yStride = blockDim.y * gridDim.y; */

  int threadRowIndex = threadIdx.y + 1;
  int threadColIndex = threadIdx.x + 1;
  /* int i = yIndex; */
  /* int j = xIndex; */

  __shared__ bool localGrid[TILE_SIZE_Y + 2][TILE_SIZE_X + 2];

  for (int i = yIndex; i <= N + 1; i += yStride)
  {
    size_t y = __umul24(i, currentGridPitch);
    size_t yNext = __umul24(i, nextGridPitch);
    size_t up = __umul24(i - 1, currentGridPitch);
    size_t down = __umul24(i + 1, currentGridPitch);

    for (int j = xIndex; j <= N + 1; j += xStride)
    {
      size_t left = j - 1;
      size_t right = j + 1;

      localGrid[threadRowIndex][threadColIndex] = currentGrid[y + j];

      if (threadRowIndex == 1)
        localGrid[threadRowIndex - 1][threadColIndex] = currentGrid[up + j];
      if (threadColIndex == 1)
        localGrid[threadRowIndex][threadColIndex - 1] = currentGrid[y  + left];
      if (threadRowIndex == TILE_SIZE_Y)
        localGrid[threadRowIndex + 1][threadColIndex] = currentGrid[down + j];
      if (threadColIndex == TILE_SIZE_X)
        localGrid[threadRowIndex][threadColIndex + 1] = currentGrid[y + right];

      if (threadRowIndex == 1 && threadColIndex == 1)
        localGrid[threadRowIndex - 1][threadColIndex - 1] = currentGrid[up + left];
      if (threadRowIndex == 1 && threadColIndex == TILE_SIZE_X)
        localGrid[threadRowIndex - 1][threadColIndex + 1] = currentGrid[up + right];
      if (threadRowIndex == TILE_SIZE_Y && threadColIndex == 1)
        localGrid[threadRowIndex + 1][threadColIndex - 1] = currentGrid[down + left];
      if (threadRowIndex == TILE_SIZE_Y && threadColIndex == TILE_SIZE_X)
        localGrid[threadRowIndex + 1][threadColIndex + 1] = currentGrid[down + right];

      __syncthreads();

      int livingNeighbors = localGrid[threadRowIndex - 1][threadColIndex - 1]
        + localGrid[threadRowIndex - 1][threadColIndex]
        + localGrid[threadRowIndex - 1][threadColIndex + 1]
        + localGrid[threadRowIndex][threadColIndex - 1]
        + localGrid[threadRowIndex][threadColIndex + 1]
        + localGrid[threadRowIndex + 1][threadColIndex - 1]
        + localGrid[threadRowIndex + 1][threadColIndex]
        + localGrid[threadRowIndex + 1][threadColIndex + 1];
      nextGrid[yNext + j] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && localGrid[threadRowIndex][threadColIndex]) ? 1 : 0;


        }
     }
      return;
  }


__device__ int sharedCalcNeighborsKernel(bool* currentGrid, size_t x, size_t left, size_t right, size_t center,
    size_t up, size_t down){
  return currentGrid[left + up] + currentGrid[x + up]
    + currentGrid[right + up] + currentGrid[left + center]
    + currentGrid[right + center] + currentGrid[left + down]
    + currentGrid[x + down] + currentGrid[right + down];
}
