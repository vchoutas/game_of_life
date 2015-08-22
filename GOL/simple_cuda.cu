#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
//#include "simple_cuda.cuh"

#define MAXBLOCKS 65535

SimpleCudaGol::SimpleCudaGoL(*int Grid,int height,int max_gen){

  this->height_= height;
  this->width_= height;
  this->maxGenerationNumber_ = max_gen;
  this->arraySize_ = height*height;

  //cudaMalloc((void**) &currentGridDevice_, arraySize_);
  cudaCheckErrors("Device memory Allocation Error!");

  cudaMalloc((void**) &nextGridDevice_, arraySize);
  cudaCheckErrors("Device memory Allocation Error!");

  if (currentGridDevice == NULL || nextGridDevice == NULL)
  {
    std::cout << "Unable to allocate Device Memory!" << std::endl;
    terminate();
  }

 cudaMemcpy(currentGridDevice,Grid, arraySize * sizeof(bool), cudaMemcpyHostToDevice);
  std::cout << "Cuda iterface initialized" << std::endl;
}


void SimpleCudaGoL::play()
{
   int x;
  //dim3 dimBlock( blocksize, blocksize );
  //dim3 dimGrid( N/dimBlock.x, N/dimBlock.y );
//
  //std::cout << "Starting Cuda playing !" << std::endl;
//
  //struct timeval startTime, endTime;
  //gettimeofday(&startTime, NULL);

  //cudaEvent_t startTimeDevice, endTimeDevice;
  /* gettimeofday(&startTime, NULL);*/
  //cudaEventCreate(&startTimeDevice);
  //cudaCheckErrors("Event Initialization Error");
  //cudaEventCreate(&endTimeDevice);
  //cudaCheckErrors("Event Initialization Error");
  //
  //cudaEventRecord(startTimeDevice, 0);y
  //tempGrid = new bool[width_ * height_];//Will be used only for final counting ;
//
  //for (int i = 0; i < maxGenerationNumber_; ++i)
  //{
    //simpleNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice, width_);
    //cudaCheckErrors("Exec Error");
    //std:swap(currentGridDevice, nextGridDevice);
  //}
  //cudaMemcpy(tempcopy, currentGridDevice, height_*height_ * sizeof(bool), cudaMemcpyDeviceToHost);
   //Copy the final grid back to the host memory.
  //cudaEventRecord(endTimeDevice, 0);
  //cudaEventSynchronize(endTimeDevice);


  //cudaEventElapsedTime(&time, startTimeDevice, endTimeDevice);
  //std::cout << "GPU Execution Time is = " << time / 1000.0f  << std::endl;
  //std::cout << " GPU Time = " << << std::endl;

//
  //cudaFree(currentGridDevice);
  //cudaFree(nextGridDevice);
  //cudaDeviceReset();
//
  //delete[] tempGrid;
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



void SimpleCudaGoL::terminate()
{
  std::cout << "Terminating cuda GOL!" << std::endl;
  delete[] currentGridDevice;
  delete[] nextGridDevice;
  std::exit(0);
  return;
}
