#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#ifdef __APPLE__
#  include <OpenGL/gl.h>
#  include <OpenGL/glu.h>
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  include <GL/freeglut.h>
#  include <GL/gl.h>
#  include <GL/glu.h>
#  include <GL/glut.h>
#endif

#define toLinearIndex(i, j, stride) (((i) * (stride)) + (j))

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
  __host__ __device__ int calcNeighbors(bool* currentGrid, int x, int left, int right, int center,
      int up, int down);

  __global__ void simpleGhostNextGenerationKernel(bool* currentGrid, bool* nextGrid, int N,
      GLubyte* colorArray);

  __global__ void multiCellGhostGridLoop(bool* currentGrid, bool* nextGrid, int N,
      GLubyte* colorArray);

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
