#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define THRESHOLD 0.3
#define GENERATIONS 10

#define SWAP(x, y) do {typeof(x) SWAP = x; x = y; y = SWAP; } while(0)
#define MIN(a, b) ((a) < (b) ? (a) : (b))


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

namespace utilities
{
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

  void read_from_file(bool *X, std::string filename, size_t N);
  bool readFile(bool *X, std::string filename, size_t N);
  void generate_table(bool *X, size_t N);
  void save_table(int *X, int N);
  int count(bool* currGrid,int height,int width,
      const std::string& prefix = std::string(""));
  int countGhost(bool* currGrid,int height,int width,
      const std::string& prefix = std::string(""));
  void generate_ghost_table(bool* Grid, bool* GhostGrid, size_t N);
  void print(bool* grid, size_t N);

} // namespace utilities

#endif
