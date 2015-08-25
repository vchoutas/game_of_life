#include <iostream>
#include <cstdlib>
#include "serial.h"
#include "simple_cuda.cuh"
#include "many_cuda.cuh"
#include "utilities.h"

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage : " << argv[0]
    << "\n\t inputGoLGridFile \n\t N(gridSize) \n\t numberOfGenerations"
    << "\n\t deviceId(optional)"<< std::endl;
    return -1;
  }

  int N = atoi(argv[2]);
  int maxGen = atoi(argv[3]);

  int deviceId = 0;
  if (argc == 5)
    deviceId = atoi(argv[4]);

  cudaSetDevice(deviceId);
  cudaCheckErrors("Cuda Device Selection Error");

  bool* startingGrid = new bool[N * N];
  if (startingGrid == NULL)
  {
    std::cout << "Could not allocate memory for the initial grid!" << std::endl;
    return -1;
  }

  //if (!utilities::readFile(startingGrid, argv[1], N))
  //{
    //std::cout << "Could not read input file!" << std::endl;
    //return -1;
  //}

  utilities::generate_table(startingGrid,N);
  // Execute the serial code.
  serial::execSerial(startingGrid, N, maxGen);

  // Execute the serial code using the ghost cells to simulate the cyclic world
  // instead of the modulus operations.
  serial::execSerialGhost(startingGrid , N, maxGen);

  // Execute the simple version of the parallel Game of Life algorithm.
  simpleCuda(startingGrid, N, maxGen);

  // Execute the simple version of the parallel Game of Life with better
  // memory allocation.
  simpleCudaPitch(startingGrid, N, maxGen);

  simpleCudaGhostPitch(startingGrid, N, maxGen);

  multiCellCudaNaive(startingGrid, N, maxGen);

  multiCellCuda(startingGrid, N, maxGen);

  delete[] startingGrid;
  return 0;
}
