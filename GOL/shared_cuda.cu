#include <iostream>

#include "shared_cuda.cuh"
#include "utilities.cuh"

#define TILE_SIZE 16
#define TILE_SIZE_X 16
#define TILE_SIZE_Y 16
#define CLP 2
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
    //cudaDeviceSynchronize();
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
  dim3 blocks(std::min(GhostN / (threadNum.x * CLP) + 1, (unsigned int)MAXBLOCKS),
      std::min(GhostN / (threadNum.y * CLP) + 1, (unsigned int)MAXBLOCKS));

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
   dim3 blocks(std::min((N  + (threadNum.x * CLP) -1) / (threadNum.x * CLP), (unsigned int)MAXBLOCKS),
      std::min((N +(threadNum.y * CLP) -1)/ (threadNum.y * CLP) , (unsigned int)MAXBLOCKS));

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
    utilities::updateGhostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice, GhostN, GhostN);
    utilities::updateGhostCorners<<< 1, 1 >>>(currentGridDevice, GhostN, GhostN);
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
    size_t nextGridPitch)
{
  size_t startPoint = __umul24(blockIdx.y , blockDim.y) * currentGridPitch +
    __umul24(blockIdx.x, blockDim.x);

  size_t row = __umul24(blockIdx.y, blockDim.y) + threadIdx.y + 1;
  size_t col = __umul24(blockIdx.x, blockDim.x) + threadIdx.x + 1;

  int i = threadIdx.y;
  int j = threadIdx.x;

  __shared__ bool localGrid[TILE_SIZE_Y + 2][TILE_SIZE_X + 2];



  int linIndex = __umul24(i, TILE_SIZE_X) + j;
  int jj = linIndex % (TILE_SIZE_X + 2);
  int ii = (linIndex - jj) / (TILE_SIZE_X + 2);
  int I = ii * currentGridPitch + jj;

  localGrid[ii][jj] = currentGrid[startPoint + I];

  int linIndex2 = __umul24(TILE_SIZE_Y, TILE_SIZE_X) + linIndex;
  int jj2 = linIndex2 % (TILE_SIZE_X + 2);
  int ii2 = (linIndex2 - jj2) / (TILE_SIZE_X + 2);

  int I2 = ii2 * currentGridPitch + jj2;

  if ((jj2 < TILE_SIZE_X + 2) && (ii2 < TILE_SIZE_Y + 2) && (I2 < (N+2) * (N+2)))
  {
    localGrid[ii2][jj2] = currentGrid[startPoint + I2];
  }

  __syncthreads();


    i++;
    j++;
    int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j]
      + localGrid[i - 1][j + 1] + localGrid[i][j - 1] + localGrid[i][j + 1]
      + localGrid[i + 1][j - 1] + localGrid[i + 1][j] + localGrid[i + 1][j + 1];
    nextGrid[row * nextGridPitch + col] = livingNeighbors == 3 ||
      (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0;

  return;
}

__global__ void multiCellSharedMemKernel(bool* currentGrid, bool* nextGrid, int N){


  size_t add1 = threadIdx.y/15;
  size_t add2 = threadIdx.x/15;

  size_t row = (blockIdx.y * blockDim.y + threadIdx.y) * CLP ;//These numbers refer to the currentGrid
  size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * CLP ;//These numbers refer to the currentGrid

  //size_t startX = blockIdx.x * blockDim.x ;
  //size_t startY = blockIdx.y * blockDim.y ;

  size_t Y = threadIdx.y*CLP ;
  size_t X = threadIdx.x*CLP ;

   __shared__ bool localGrid[TILE_SIZE*CLP + 2][TILE_SIZE*CLP + 2];


  //COPY THE INTERNAL PARTS
  for (size_t i = 0; i < CLP+1; i++){

    size_t y = (row +add1 + i ) * (N + 2);//The global grid has a N+2 edge
    for (size_t j = 0; j < CLP+1; j++)

    {
      size_t x = col + j +add2;
      localGrid[ Y +add1 +i ][X+add2 +j ] = currentGrid[y + x];//WE add +1 in order to fill the center parts
    }
  }

  __syncthreads();

  int i,j;


  for (size_t m = 1; m < CLP+1 ; m++){
      i = Y + m;
      size_t y = __umul24(row + m , N + 2);
      for (size_t n = 1; n < CLP+1 ; n++){
        j = X + n;
        int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j]
            + localGrid[i - 1][j + 1] + localGrid[i][j - 1]
            + localGrid[i][j + 1] + localGrid[i + 1][j - 1] + localGrid[i + 1][j]
            + localGrid[i + 1][j + 1];

          size_t x = col + n ;

          nextGrid[y + x] = livingNeighbors == 3 ||
            (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0;
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


  int threadRowIndex = threadIdx.y + 1;
  int threadColIndex = threadIdx.x + 1;

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
      {
        localGrid[threadRowIndex - 1][threadColIndex] = currentGrid[up + j];
        localGrid[threadColIndex][threadRowIndex - 1] =
          currentGrid[(i - threadIdx.y + threadIdx.x) * currentGridPitch - 1];
      }
        /* localGrid[threadRowIndex - 1][threadColIndex] = currentGrid[up + j]; */
      /* if (threadColIndex == 1) */
        /* localGrid[threadRowIndex][threadColIndex - 1] = currentGrid[y  + left]; */
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
