#include "cuda_kernels.cuh"

namespace cuda_kernels
{
  /**
   * @brief Function used to calculate the number of neighbors of a cell
   * @param currentGrid[bool*] The current Game of Life board.
   * @param centerCol[int] The column of the current cell.
   * @param leftCol[int] The column on the left of the current cell.
   * @param rightCol[int] The column on the right of the current cell.
   * @param centerRow[int] The row where the current cell is situated.
   * @param topRow[int] The row above the current cell.
   * @param bottomRow[int] The row below the current cell.
   * @return int The number of living neighbors of the cell located in
   *          x, y = centerCol, centerRow
   */
  __host__ __device__ int calcNeighbors(bool* currentGrid, int centerCol, int leftCol,
      int rightCol, int centerRow, int topRow, int bottomRow)
  {
    return currentGrid[leftCol + topRow] + currentGrid[centerCol + topRow]
      + currentGrid[rightCol + topRow] + currentGrid[leftCol + centerRow]
      + currentGrid[rightCol + centerRow] + currentGrid[leftCol + bottomRow]
      + currentGrid[centerCol + bottomRow] + currentGrid[rightCol + bottomRow];
  }

  /**
   * @brief A simple CUDA kernel used to calculate the next iteration of the
   * Game of Life.
   * @param currentGrid[bool*] The current board.
   * @param nextGrid[bool*] The board of the next generation of the game.
   * @param N[int] The number of cells in each row.
   * @param colorArray[GLubyte*] The array that contains the color of each cell.
   * @return void
   */
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

  /**
   * @brief A CUDA kernel that uses a 2D grid size loop to calculate the next
   * generation of the Game of Life.
   * @param currentGrid[bool*] The current board.
   * @param nextGrid[bool*] The board of the next generation of the game.
   * @param N[int] The number of cells in each row.
   * @param colorArray[GLubyte*] The array that contains the color of each cell.
   * @return void
   */
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
   * @brief A CUDA kernel that uses shared memory tiles to speed up the next gen computation.
   * @param currentGrid[bool*] The current board.
   * @param nextGrid[bool*] The board of the next generation of the game.
   * @param N[int] The number of cells in each row.
   * @param colorArray[GLubyte*] The array that contains the color of each cell.
   * @return void
   */
  __global__ void sharedMemoryKernel(bool* currentGrid, bool* nextGrid, int N, GLubyte* colorArray)
  {
    size_t add1 = threadIdx.y / (TILE_SIZE_Y - 1);
    size_t add2 = threadIdx.x / (TILE_SIZE_X - 1);

    size_t row = (blockIdx.y * blockDim.y + threadIdx.y) * CELLS_PER_THREAD ;//These numbers refer to the currentGrid
    size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * CELLS_PER_THREAD ;//These numbers refer to the currentGrid

    //size_t startX = blockIdx.x * blockDim.x ;
    //size_t startY = blockIdx.y * blockDim.y ;

    size_t Y = threadIdx.y * CELLS_PER_THREAD;
    size_t X = threadIdx.x * CELLS_PER_THREAD;

    __shared__ bool localGrid[TILE_SIZE_Y * CELLS_PER_THREAD + 2][TILE_SIZE_X * CELLS_PER_THREAD + 2];

    //COPY THE INTERNAL PARTS
    for (size_t i = 0; i < CELLS_PER_THREAD + 1; i++){

      size_t y = (row + add1 + i ) * (N + 2);//The global grid has a N+2 edge
      for (size_t j = 0; j < CELLS_PER_THREAD + 1; j++)

      {
        size_t x = col + j + add2;
        localGrid[Y + add1 + i][X + add2 + j] = currentGrid[y + x];//WE add +1 in order to fill the center parts
      }
    }

    __syncthreads();

    int i,j;
    for (size_t m = 1; m < CELLS_PER_THREAD + 1; m++)
    {
      i = Y + m;
      size_t y = __umul24(row + m, N + 2);
      for (size_t n = 1; n < CELLS_PER_THREAD + 1; n++)
      {
        j = X + n;
        int livingNeighbors = localGrid[i - 1][j - 1] + localGrid[i - 1][j]
          + localGrid[i - 1][j + 1] + localGrid[i][j - 1]
          + localGrid[i][j + 1] + localGrid[i + 1][j - 1] + localGrid[i + 1][j]
          + localGrid[i + 1][j + 1];

        size_t x = col + n;

        nextGrid[y + x] = livingNeighbors == 3 ||
          (livingNeighbors == 2 && localGrid[i][j]) ? 1 : 0;

        int index = y + x;
        int colorIndex = 3 * ((row + m - 1) * N  + x - 1);
        if (nextGrid[index] && !localGrid[i][j])
        {
          colorArray[colorIndex]  = 0;
          colorArray[colorIndex + 1]  = 255;
          // colorArray[colorIndex + 2]  = 0;
        }
        // If the cell was alive and died.
        if (!nextGrid[index] && localGrid[i][j])
        {
          colorArray[colorIndex] = 255;
          colorArray[colorIndex + 1] = 0;
          // colorArray[colorIndex + 2] = 0;
        }
        else if(!nextGrid[index] && !localGrid[i][j])
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
