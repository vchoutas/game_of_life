#ifndef SERIAL_H
#define SERIAL_H

namespace serial
{
  
  int calcNeighbors(bool* currGrid,int x, int left, int right, int center,
    int up, int down);
  void getNextGeneration(bool* currGrid,bool* nextGrid,int height,int width);
  
  }//namespace serial

#endif
