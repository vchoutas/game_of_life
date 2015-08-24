#include <sys/time.h>
#include "serial.h"

namespace serial
{

//Na to kanouem makro na teleiwnoume
//GIRD[y*(N)+x] is the the value at GRID(x,y)
void createGhostCells(bool* Grid,int N)
{
  for (int x = 1; x < N-1; ++x)  {//The first and last columns are to be wrriten
    Grid[x + (N-1) * N ] = Grid[ x +    1 * N   ];//write bottom to top
    Grid[x +     0 * N ] = Grid[ x + (N-2)*N];//write top to bottom
    }
for ( int y = 1; y <N-1;++y){
    Grid[y*N + N-1] = Grid[y*N +  1 ];//write left  to   right
    Grid[y*N +  0 ] = Grid[y*N + N-2];//write right  to left
  }

  //write the corners
  Grid[0*N     +  0 ] = Grid[ (N-2)*N + N-2];//(0,0)-->(N-2,N-2)
  Grid[(N-1)*N + N-1] = Grid[ 1*N     + 1 ];//(N-1,N-1)-->(1,1)
  Grid[0*N     + N-1] = Grid[ (N-2)*N + 1];//(0,N-1)-->(N-2,1)
  Grid[(N-1)*N +  0 ] = Grid[ 1*N     + N-2];//(N-1,0)-->(1,N-2)

  return;
}


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

void getNextGenerationGhost(bool* currGrid,bool* nextGrid,int N)
{
  for (int y = 1; y < N+1; ++y)
  {
    int up =  (y - 1) * (N+2);
    int center = y * (N+2);
    int down = (y + 1) * (N+2);
    for (int x = 1; x < N+1; ++x)
    {
      int left  = (x - 1) ;
      int right = (x + 1);

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

void execSerial(bool** startingGrid, bool** finalGrid, int N, int maxGen)
{
  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);
  for (int i = 0; i < maxGen; ++i)
  {
    getNextGeneration(*startingGrid, *finalGrid, N, N);
    SWAP(*startingGrid, *finalGrid);
  }
  // Swap the pointers so that the final table is in the finalGrid pointer.
  SWAP(*startingGrid, *finalGrid);
  gettimeofday(&endTime, NULL);

  std::string prefix("[Serial Game of Life]: ");
  double serialExecTime = (double)((endTime.tv_usec - startTime.tv_usec)
      /1.0e6 + endTime.tv_sec - startTime.tv_sec);
  std::cout << std::endl << prefix << "<" << serialExecTime << "> seconds" << std::endl;
  utilities::count(*finalGrid, N, N, prefix);

  return;
}


void execSerialGhost(bool** startingGrid, bool** finalGrid, int N, int maxGen)
{
  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);
  for (int i = 0; i < maxGen; ++i)
  {
    createGhostCells(*startingGrid,N+2);
    getNextGenerationGhost(*startingGrid, *finalGrid, N);
    SWAP(*startingGrid, *finalGrid);
  }
  // Swap the pointers so that the final table is in the finalGrid pointer.
  SWAP(*startingGrid, *finalGrid);
  gettimeofday(&endTime, NULL);

  double serialExecTime = (double)((endTime.tv_usec - startTime.tv_usec)
      /1.0e6 + endTime.tv_sec - startTime.tv_sec);
  std::cout << "[Ghosts Serial Game of Life]: <" << serialExecTime << "> seconds" << std::endl;

  return;
}


} // namespace serial
