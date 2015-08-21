#include "serial.h"



namespace serial{

void getNextGeneration(bool* currGrid,bool* nextGrid,int height,int width)
{
  for (int y = 0; y < height; ++y)
  {
    size_t up = ( (y + height - 1) % height) * width;
    size_t center = y * width;
    size_t down = ((y + 1) % height) * width;
    for (int x = 0; x < width; ++x)
    {
      size_t left = (x + width - 1) % width;
      size_t right = (x + 1) % width;

      int livingNeighbors = serial::calcNeighbors(currGrid ,x, left, right,center,up,down);
      nextGrid[center + x] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && currGrid[x + center]) ? 1 : 0;
    }
  }

  return;
}

int calcNeighbors(bool* currGrid,int x, int left, int right, int center,
    int up, int down)
{
  return currGrid[left + up] + currGrid[x + up]
      + currGrid[right + up] + currGrid[left + center]
      + currGrid[right + center] + currGrid[left + down]
      + currGrid[x + down] + currGrid[right + down];
}

} // namespace utilities
