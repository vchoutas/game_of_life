#ifndef SIMPLE_CUDA_GOL_H
#define SIMPLE_CUDA_GOL_H
#include "utilities.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void simpleNextGenerationKernel(bool* currentGrid, bool* nextGrid, const int N);
__device__ inline int calcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center, int up , int down);

class SimpleCudaGoL
{
  public:
    // Default Constructor.
    SimpleCudaGoL(){}
    // Constructor that creates a random square grid
    SimpleCudaGoL(*int Grid,int height,int max_gen);
    ~SimpleCudaGoL()
    {
      std::cout << "Destroying Simple Cuda Game of Life Object!" << std::endl;
      delete[] currentGrid_;
      delete[] nextGrid_;
      void play();

    // Function to initialize the necessary OpenGL components.
    void play();

  private:
    bool* currentGridDevice_;
    bool* nextGridDevice_;
    size_t arraySize_ ;
    int width_;
    int height_;
    int maxGenerationNumber_;

};

#endif // SIMPLE_CUDA_GOL_H
