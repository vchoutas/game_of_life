#ifndef SERIAL_H
#define SERIAL_H

#include "utilities.h"

namespace serial
{
  int calcNeighbors(bool* currGrid,int x, int left, int right, int center,
    int up, int down);
  void getNextGeneration(bool* currGrid,bool* nextGrid,int height,int width);
  void execSerial(bool* startingGrid, bool* finalGrid, int N, int maxGen);

}  // namespace serial

#endif
