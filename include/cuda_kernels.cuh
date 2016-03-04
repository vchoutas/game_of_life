#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#ifdef __APPLE__
#  include <OpenGL/gl.h>
#else
#  include <GL/gl.h>
#endif

#define toLinearIndex(i, j, stride) (((i) * (stride)) + (j))

#define CELLS_PER_THREAD 2
#define TILE_SIZE_X 16
#define TILE_SIZE_Y 16

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
          msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

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
      int rightCol, int centerRow, int topRow, int bottomRow);

  /**
   * @brief A simple CUDA kernel used to calculate the next iteration of the
   * Game of Life.
   * @param currentGrid[bool*] The current board.
   * @param nextGrid[bool*] The board of the next generation of the game.
   * @param N[int] The number of cells in each row.
   * @return void
   */
  __global__ void simpleGhostNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N);

  /**
   * @brief A CUDA kernel that uses a 2D grid size loop to calculate the next
   * generation of the Game of Life.
   * @param currentGrid[bool*] The current board.
   * @param nextGrid[bool*] The board of the next generation of the game.
   * @param N[int] The number of cells in each row.
   * @return void
   */
  __global__ void multiCellGhostGridLoop(bool* currentGrid, bool* nextGrid, int N);

  /**
   * @brief A CUDA kernel that uses shared memory tiles to speed up the next gen computation.
   * @param currentGrid[bool*] The current board.
   * @param nextGrid[bool*] The board of the next generation of the game.
   * @param N[int] The number of cells in each row.
   * @return void
   */
  __global__ void sharedMemoryKernel(bool* currentGrid, bool* nextGrid, int N);


  __global__ void updateColorArray(GLubyte* colorArray, bool* currentGrid, bool* nextGrid, int N);

  /**
   * @brief Updates the elements of the extra ghost rows.
   * @param grid[bool*] The square array that will be updated
   * @param N[int] The size of the array including the extra rows and columns.
   * @param pitch[size_t] The stride used to access the next row of the array.
   * @return void
   */
  __global__ void updateGhostRows(bool* grid, int N, size_t pitch);

  /**
   * @brief Updates the elements of the extra ghost columns.
   * @param grid[bool*] The square array that will be updated
   * @param N[int] The size of the array including the extra rows and columns.
   * @param pitch[size_t] The stride used to access the next row of the array.
   * @return void
   */
  __global__ void updateGhostCols(bool* grid, int N, int pitch);

  /**
   * @brief Updates the corners of the array.
   * @param grid[bool*] The square array that will be updated
   * @param N[int] The size of the array including the extra rows and columns.
   * @param pitch[size_t] The stride used to access the next row of the array.
   * @return void
   */
  __global__ void updateGhostCorners(bool* grid, int N, int pitch);

}  // namespace cuda_kernels

#endif  // CUDA_KERNELS_CUH
