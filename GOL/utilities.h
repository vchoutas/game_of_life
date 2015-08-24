#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstdio>
#include <iostream>

#define THRESHOLD 0.3
#define GENERATIONS 10

#define SWAP(x, y) do {typeof(x) SWAP = x; x = y; y = SWAP; } while(0)

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
  void read_from_file(bool *X, std::string filename, size_t N);
  bool readFile(bool *X, std::string filename, size_t N);
  void generate_table(bool *X, size_t N);
  void save_table(int *X, int N);
  int count(bool* currGrid,int height,int width,
      const std::string& prefix=std::string(""));
  int countGhost(bool* currGrid,int height,int width);
  void generate_ghost_table(bool *Grid,bool *GhostGrid, size_t N);
  void print(bool* grid,size_t N );

} // namespace utilities

#endif
