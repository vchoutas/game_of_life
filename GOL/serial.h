#ifndef SERIAL_H
#define SERIAL_H

#include "utilities.h"

namespace serial
{
  int calcNeighbors(bool* currGrid,int x, int left, int right, int center,
    int up, int down);
  void getNextGeneration(bool* currGrid,bool* nextGrid,int height,int width);
  void  getNextGenerationGhost(bool* currGrid,bool* nextGrid,int N);
  void execSerial(bool** startingGrid, bool** finalGrid, int N, int maxGen);
  void execSerialGhost(bool** serialStartingGhostGrid ,bool** finalSerialGhostGrid,  int N,int  maxGen);
  void createGhostCells(bool* Grid,int N);

}  // namespace serial

#endif
