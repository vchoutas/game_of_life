#include <sys/time.h>
#include "serial.h"

namespace serial
{

void getNextGeneration(bool* currGrid,bool* nextGrid,int height,int width)
{
  for (int y = 0; y < height; ++y)
  {
    int up = ( (y + height - 1) % height) * width;
    int center = y * width;
    int down = ((y + 1) % height) * width;
    for (int x = 0; x < width; ++x)
    {
      int left = (x + width - 1) % width;
      int right = (x + 1) % width;

      int livingNeighbors = serial::calcNeighbors(currGrid ,x, left, right,center,up,down);
      nextGrid[center + x] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && currGrid[x + center]) ? 1 : 0;
    }
  }

  return;
}

int calcNeighbors(bool* currGrid, int x, int left, int right, int center,
    int up, int down)
{
  return currGrid[left + up] + currGrid[x + up]
      + currGrid[right + up] + currGrid[left + center]
      + currGrid[right + center] + currGrid[left + down]
      + currGrid[x + down] + currGrid[right + down];
}

void execSerial(bool* startingGrid, bool* finalGrid, int N, int maxGen)
{
  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);
  for (int i = 0; i < maxGen; ++i)
  {
    getNextGeneration(startingGrid, finalGrid, N, N);
    SWAP(startingGrid, finalGrid);
  }
  // Swap the pointers so that the final table is in the finalGrid pointer.
  SWAP(startingGrid, finalGrid);
  gettimeofday(&endTime, NULL);

  double serialExecTime = (double)((endTime.tv_usec - startTime.tv_usec)
      /1.0e6 + endTime.tv_sec - startTime.tv_sec);
  std::cout << "[Serial Game of Life]: <" << serialExecTime << "> seconds" << std::endl;

  return;
}

} // namespace serial
