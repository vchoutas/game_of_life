#include "cuda_kernels.cuh"

namespace cuda_kernels
{
  __host__ __device__ int calcNeighbors(bool* currentGrid, int x, int left, int right, int center,
      int up, int down)
  {
    return currentGrid[left + up] + currentGrid[x + up]
      + currentGrid[right + up] + currentGrid[left + center]
      + currentGrid[right + center] + currentGrid[left + down]
      + currentGrid[x + down] + currentGrid[right + down];
  }
  __global__ void simpleGhostNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N,
      GLubyte* colorArray)
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

      int index = center + col;

      int livingNeighbors = calcNeighbors(currentGrid, col, left, right, center, up, down);
      nextGrid[index] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && currentGrid[index]) ? 1 : 0;

      int colorIndex = 3 * ((row - 1) * N + col - 1);
      if (nextGrid[index] && !currentGrid[index])
      {
        colorArray[colorIndex]  = 0;
        colorArray[colorIndex + 1]  = 255;
        // colorArray[colorIndex + 2]  = 0;
      }
      // If the cell was alive and died.
      if (!nextGrid[index] && currentGrid[index])
      {
        colorArray[colorIndex] = 255;
        colorArray[colorIndex + 1] = 0;
        // colorArray[colorIndex + 2] = 0;
      }
      else if(!nextGrid[index] && !currentGrid[index])
      {
        colorArray[colorIndex] > 0 ? colorArray[colorIndex]-- : colorArray[colorIndex] = 0;
      }

      if (nextGrid[index])
      {
        colorArray[colorIndex + 2] >= 255 ? colorArray[colorIndex + 2] = 255:
          colorArray[colorIndex + 2]++;
      }

    }
    return;
  }

  __global__ void multiCellGhostGridLoop(bool* currentGrid, bool* nextGrid, int N,
      GLubyte* colorArray)
  {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int xStride = __umul24(blockDim.x, gridDim.x);
    int yStride = __umul24(blockDim.y, gridDim.y);

    for (int i = yIndex; i < N + 1; i += yStride)
    {
      size_t y = __umul24(i, N + 2);
      size_t up = __umul24(i - 1, N + 2);
      size_t down = __umul24(i + 1, N + 2);
      for (int j = xIndex; j < N + 1; j += xStride)
      {
        size_t left = j - 1;
        size_t right = j + 1;

        int index = y + j;

        int livingNeighbors = calcNeighbors(currentGrid, j, left, right, y, up, down);
        nextGrid[index] = livingNeighbors == 3 ||
          (livingNeighbors == 2 && currentGrid[index]) ? 1 : 0;

        int colorIndex = 3 * ((i - 1) * N  + j - 1);
        if (nextGrid[index] && !currentGrid[index])
        {
          colorArray[colorIndex]  = 0;
          colorArray[colorIndex + 1]  = 255;
          // colorArray[colorIndex + 2]  = 0;
        }
        // If the cell was alive and died.
        if (!nextGrid[index] && currentGrid[index])
        {
          colorArray[colorIndex] = 255;
          colorArray[colorIndex + 1] = 0;
          // colorArray[colorIndex + 2] = 0;
        }
        else if(!nextGrid[index] && !currentGrid[index])
        {
          colorArray[colorIndex] > 0 ? colorArray[colorIndex]-- : colorArray[colorIndex] = 0;
        }

        if (nextGrid[index])
        {
          colorArray[colorIndex + 2] >= 255 ? colorArray[colorIndex + 2] = 255:
            colorArray[colorIndex + 2]++;
        }
      }
    }
    return;
  }

  /**
   * @brief Updates the elements of the extra ghost rows.
   * @param grid[bool*] The square array that will be updated
   * @param N[int] The size of the array including the extra rows and columns.
   * @param pitch[size_t] The stride used to access the next row of the array.
   * @return void
   */
  __global__ void updateGhostRows(bool* grid, int N, size_t pitch)
  {
    int x = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (x < N - 1)
    {
      //The first and last columns are to be wrriten
      grid[toLinearIndex(N - 1, x, pitch)] = grid[toLinearIndex(1, x, pitch)];  //write bottom to top
      grid[toLinearIndex(0, x, pitch)] = grid[toLinearIndex(N - 2, x, pitch)];  //write top to bottom
    }
  }

  /**
   * @brief Updates the elements of the extra ghost columns.
   * @param grid[bool*] The square array that will be updated
   * @param N[int] The size of the array including the extra rows and columns.
   * @param pitch[size_t] The stride used to access the next row of the array.
   * @return void
   */
  __global__ void updateGhostCols(bool* grid, int N, int pitch)
  {
    int y = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (y < N-1)
    {
      //std:cout<<id;
      grid[toLinearIndex(y, N - 1, pitch)] = grid[toLinearIndex(y, 1, pitch)];  //write left  to   right
      grid[toLinearIndex(y, 0, pitch)] = grid[toLinearIndex(y, N - 2, pitch)];  //write right  to left
    }
  }
  /**
   * @brief Updates the corners of the array.
   * @param grid[bool*] The square array that will be updated
   * @param N[int] The size of the array including the extra rows and columns.
   * @param pitch[size_t] The stride used to access the next row of the array.
   * @return void
   */
  __global__ void updateGhostCorners(bool* grid, int N, int pitch)
  {
    grid[toLinearIndex(0, 0, pitch)] = grid[toLinearIndex(N-2, N - 2, pitch)];//(0,0)-->(N-2,N-2)
    grid[toLinearIndex(N-1, N - 1, pitch)] = grid[toLinearIndex(1, 1, pitch)];//(N-1,N-1)-->(1,1)
    grid[toLinearIndex(0, N - 1, pitch)] = grid[toLinearIndex(N - 2, 1, pitch)];//(0,N-1)-->(N-2,1)
    grid[toLinearIndex(N - 1, 0, pitch)] = grid[toLinearIndex(1, N - 2, pitch)];//(N-1,0)-->(1,N-2)
  }

}  // namespace cuda_kernels
