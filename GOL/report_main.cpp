#include <iostream>
#include <cstdlib>
#include <cstring>
#include "serial.h"
#include "simple_cuda.cuh"
#include "many_cuda.cuh"

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
  bool* finalSerialGrid = new bool[N * N];
  if (finalSerialGrid == NULL)
  {
    std::cout << "Could not allocate memory for the final grid for the serial code!" << std::endl;
    delete[] startingGrid;
    return -1;
  }

  if (!utilities::readFile(startingGrid, argv[1], N))
  {
    std::cout << "Could not read input file!" << std::endl;
    return -1;
  }

  bool* serialStartingGrid = new bool[N * N];
  if (serialStartingGrid == NULL)
  {
    std::cout << "Could not allocate memory for the initial grid of the serial code!" << std::endl;
  }
  memcpy(serialStartingGrid, startingGrid, N * N * sizeof(bool));
  serial::execSerial(&serialStartingGrid, &finalSerialGrid, N, maxGen);

  bool* simpleGpuStartingGrid = new bool[N * N];
  bool* simpleGpuFinalGrid = new bool[N * N];
  if (simpleGpuStartingGrid == NULL)
  {
    std::cout << "Could not allocate memory for the initial grid array(simple gpu version)!" << std::endl;
    return -1;
  }
  if (simpleGpuFinalGrid == NULL)
  {
    std::cout << "Could not allocate memory for the final grid array(simple gpu version)!" << std::endl;
    return -1;
  }
  memcpy(simpleGpuStartingGrid, startingGrid, N * N * sizeof(bool));
  simple_cuda(&simpleGpuStartingGrid, &simpleGpuFinalGrid, N, maxGen);

  bool* simpleGpuPitchStartingGrid = new bool[N * N];
  bool* simpleGpuPitchFinalGrid = new bool[N * N];
  if (simpleGpuPitchStartingGrid == NULL)
  {
    std::cout << "Could not allocate memory for the initial grid array(simple gpu version)!" << std::endl;
    return -1;
  }
  if (simpleGpuPitchFinalGrid == NULL)
  {
    std::cout << "Could not allocate memory for the final grid array(simple gpu version)!" << std::endl;
    return -1;
  }
  memcpy(simpleGpuPitchStartingGrid, startingGrid, N * N * sizeof(bool));
  simpleCudaPitch(&simpleGpuPitchStartingGrid, &simpleGpuPitchFinalGrid, N, maxGen);

  bool* manyGpuStartingGrid = new bool[N * N];
  bool* manyGpuFinalGrid = new bool[N * N];
  if (manyGpuStartingGrid == NULL)
  {
    std::cout << "Could not allocate memory for the initial grid array(many gpu version)!" << std::endl;
    return -1;
  }
  if (manyGpuFinalGrid == NULL)
  {
    std::cout << "Could not allocate memory for the final grid array(many gpu version)!" << std::endl;
    return -1;
  }
  memcpy(manyGpuStartingGrid, startingGrid, N * N * sizeof(bool));
  many_cuda(&manyGpuStartingGrid, &manyGpuFinalGrid, N, maxGen);

  delete[] startingGrid;
  delete[] serialStartingGrid;
  delete[] finalSerialGrid;
  delete[] simpleGpuStartingGrid;
  delete[] simpleGpuFinalGrid;
  delete[] simpleGpuPitchStartingGrid;
  delete[] simpleGpuPitchFinalGrid;
  delete[] manyGpuStartingGrid;
  delete[] manyGpuFinalGrid;
  return 0;
}
