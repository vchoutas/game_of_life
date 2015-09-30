#include <sys/time.h>
#include "serial.h"

namespace serial
{
  void createGhostCells(bool* Grid,int N)
  {
    for (int x = 1; x < N - 1; ++x)
    {
      //The first and last columns are to be wrriten
      Grid[toLinearIndex(N - 1, x, N)] = Grid[toLinearIndex(1, x, N)];  //write bottom to top
      Grid[toLinearIndex(0, x, N)] = Grid[toLinearIndex(N - 2, x, N)];  //write top to bottom
    }
    for ( int y = 1; y < N - 1; ++y)
    {
      Grid[toLinearIndex(y, N - 1, N)] = Grid[toLinearIndex(y, 1, N)];  //write left  to   right
      Grid[toLinearIndex(y, 0, N)] = Grid[toLinearIndex(y, N - 2, N)];  //write right  to left
    }

    //write the corners
    Grid[toLinearIndex(0, 0, N)] = Grid[toLinearIndex(N - 2, N - 2, N)];//(0,0)-->(N-2,N-2)
    Grid[toLinearIndex(N - 1, N - 1, N)] = Grid[toLinearIndex(1, 1, N)];//(N-1,N-1)-->(1,1)
    Grid[toLinearIndex(0, N - 1 , N)] = Grid[toLinearIndex(N - 2, 1, N)];//(0,N-1)-->(N-2,1)
    Grid[toLinearIndex(N - 1, 0, N)] = Grid[toLinearIndex(1, N - 2, N)];//(N-1,0)-->(1,N-2)

    return;
  }


  void getNextGeneration(bool* currGrid,bool* nextGrid,int height,int width)
  {
    for (int y = 0; y < height; ++y)
    {
      int up = ((y + height - 1) % height) * width;
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

  void execSerial(bool* startingGrid, int N, int maxGen)
  {

    std::string prefix("[Serial Game of Life]: ");
    // The pointer to the Game of Life grid after maxGen generations.
    bool* nextGameGrid = new bool[N * N];
    bool* initialGameGrid = new bool[N * N];
    if (initialGameGrid == NULL)
    {
      std::cout << prefix << "Could not allocate memory for the initial grid!" << std::endl;
    }

    if (nextGameGrid == NULL)
    {
      std::cout << prefix << "Could not allocate memory for the final grid!" << std::endl;
      delete[] startingGrid;
      return;
    }

    memcpy(initialGameGrid, startingGrid, N * N * sizeof(bool));

    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    for (int i = 0; i < maxGen; ++i)
    {
      getNextGeneration(initialGameGrid, nextGameGrid, N, N);
      SWAP(initialGameGrid, nextGameGrid);
    }
    // Swap the pointers so that the final table is in the nextGameGrid pointer.
    SWAP(initialGameGrid, nextGameGrid);
    gettimeofday(&endTime, NULL);

    double serialExecTime = (double)((endTime.tv_usec - startTime.tv_usec)
        /1.0e6 + endTime.tv_sec - startTime.tv_sec);
    std::cout << std::endl << prefix << "Execution Time is <" << serialExecTime << "> seconds" << std::endl;
    utilities::count(nextGameGrid, N, N, prefix);

    delete[] initialGameGrid;
    delete[] nextGameGrid;

    return;
  }


  void execSerialGhost(bool* startingGrid, int N, int maxGen)
  {
    std::string prefix("[Ghosts Serial Game of Life]: ");

    bool* initialGameGrid = new bool[(N + 2) * (N + 2)];
    if (initialGameGrid == NULL)
    {
      std::cout << prefix << "Could not allocate memory for the initial grid!"
        << std::endl;
      return;
    }

    bool* nextGameGrid = new bool[(N + 2)* (N + 2)];
    if (nextGameGrid == NULL)
    {
      std::cout << prefix << "Could not allocate memory for the final grid!" << std::endl;
      delete[] initialGameGrid;
      return;
    }

    utilities::generate_ghost_table(startingGrid, initialGameGrid, N);

    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    for (int i = 0; i < maxGen; ++i)
    {
      createGhostCells(initialGameGrid, N + 2);
      getNextGenerationGhost(initialGameGrid, nextGameGrid, N);
      SWAP(initialGameGrid, nextGameGrid);
    }
    // Swap the pointers so that the final table is in the finalGrid pointer.
    SWAP(initialGameGrid, nextGameGrid);
    gettimeofday(&endTime, NULL);

    double serialExecTime = (double)((endTime.tv_usec - startTime.tv_usec)
        /1.0e6 + endTime.tv_sec - startTime.tv_sec);
    std::cout << std::endl << prefix << "<" << serialExecTime << "> seconds" << std::endl;
    utilities::countGhost(nextGameGrid, N, N, prefix);

    delete[] initialGameGrid;
    delete[] nextGameGrid;
    return;
  }


} // namespace serial
