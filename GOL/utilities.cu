#include "utilities.cuh"

namespace utilities
{
  __global__ void ghostRows(bool* Grid, int N)//Does not  copy corners twp
  {
    int x = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (x < N - 1)
    {
      //The first and last columns are to be wrriten
      Grid[toLinearIndex(N - 1, x, N)] = Grid[toLinearIndex(1, x, N)];  //write bottom to top
      Grid[toLinearIndex(0, x, N)] = Grid[toLinearIndex(N - 2, x, N)];  //write top to bottom
    }
  }

  __global__ void ghostRowsPitch(bool* Grid, int N, size_t pitch)//Does not  copy corners twp
  {
    int x = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (x < N - 1)
    {
      //The first and last columns are to be wrriten
      Grid[toLinearIndex(N - 1, x, pitch)] = Grid[toLinearIndex(1, x, pitch)];  //write bottom to top
      Grid[toLinearIndex(0, x, pitch)] = Grid[toLinearIndex(N - 2, x, pitch)];  //write top to bottom
    }
  }

  __global__ void ghostCols(bool* Grid,int N)//Does not copy corners
  {
    int y = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (y< N-1)
    {
      //std:cout<<id;
      Grid[toLinearIndex(y, N - 1, N)] = Grid[toLinearIndex(y, 1, N)];  //write left  to   right
      Grid[toLinearIndex(y, 0, N)] = Grid[toLinearIndex(y, N - 2, N)];  //write right  to left

    }
  }

  __global__ void ghostColsPitch(bool* Grid, int N, int pitch)//Does not copy corners
  {
    int y = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (y< N-1)
    {
      //std:cout<<id;
      Grid[toLinearIndex(y, N - 1, pitch)] = Grid[toLinearIndex(y, 1, pitch)];  //write left  to   right
      Grid[toLinearIndex(y, 0, pitch)] = Grid[toLinearIndex(y, N - 2, pitch)];  //write right  to left

    }
  }

  __global__ void ghostCorners(bool* grid, int N)
  {
    grid[toLinearIndex(0, 0, N)] = grid[toLinearIndex(N-2, N - 2, N)];//(0,0)-->(N-2,N-2)
    grid[toLinearIndex(N-1, N - 1, N)] = grid[toLinearIndex(1, 1, N)];//(N-1,N-1)-->(1,1)
    grid[toLinearIndex(0, N - 1, N)] = grid[toLinearIndex(N - 2, 1, N)];//(0,N-1)-->(N-2,1)
    grid[toLinearIndex(N - 1, 0, N)] = grid[toLinearIndex(1, N - 2, N)];//(N-1,0)-->(1,N-2)
  }

  __global__ void ghostCornersPitch(bool* grid, int N, int pitch)
  {
    grid[toLinearIndex(0, 0, pitch)] = grid[toLinearIndex(N-2, N - 2, pitch)];//(0,0)-->(N-2,N-2)
    grid[toLinearIndex(N-1, N - 1, pitch)] = grid[toLinearIndex(1, 1, pitch)];//(N-1,N-1)-->(1,1)
    grid[toLinearIndex(0, N - 1, pitch)] = grid[toLinearIndex(N - 2, 1, pitch)];//(0,N-1)-->(N-2,1)
    grid[toLinearIndex(N - 1, 0, pitch)] = grid[toLinearIndex(1, N - 2, pitch)];//(N-1,0)-->(1,N-2)
  }

  void read_from_file(bool *X, std::string filename, size_t N){

    FILE *fp = fopen(filename.c_str(), "r+");
    if (fp == NULL)
    {
      std::cout << "Could not open file " << filename << std::endl;
      std::cout << "Exiting!" << std::endl;
      std::exit(-1);
    }

    int size = fread(X, sizeof(bool), N*N, fp);

    std::cout << "elements: " <<  size << std::endl;

    fclose(fp);

  }

  bool readFile(bool *X, std::string filename, size_t N)
  {
    FILE *fp = fopen(filename.c_str(), "r+");
    if (fp == NULL)
    {
      printf("Invalid File Name!\n");
      return false;
    }

    int size = fread(X, sizeof(bool), N * N, fp);

    printf("Elements: %d\n", size);

    fclose(fp);
    return true;
  }

  void generate_table(bool *X, size_t N){
    srand(time(NULL));
    int counter = 0;

    for(size_t i=0; i<N; i++){
      for(size_t j=0; j<N; j++){
        X[ i * N + j] = ( (float)rand() / (float)RAND_MAX ) < THRESHOLD;
        counter += X[toLinearIndex(i, j, N)];
      }
    }

    std::cout << "Number of non zerow elements: " << counter << std::endl;
    std::cout << "Percent: " << (float)counter / (float)(N*N) << std::endl;
  }

  void generate_ghost_table(bool* Grid, bool* GhostGrid, size_t N){

    for(size_t i=0; i<N+2; i++){
      for(size_t j=0; j<N+2; j++){
        GhostGrid[toLinearIndex(i, j, N+2)] = 0;
      }
    }
    for(size_t i=0; i<N; i++){
      for(size_t j=0; j<N; j++){
        GhostGrid[toLinearIndex(i + 1, j + 1, N + 2)] = Grid[toLinearIndex(i, j, N)];
      }
    }

    //std::cout << "Ghost grid  created" << std::endl;
  }

  void save_table(bool *X, size_t N){

    FILE *fp;

    char filename[20];

    std::cout << filename << "table " << N << "x" << N << std::endl;

    std::cout <<"Saving table in file " << filename << std::endl;

    fp = fopen(filename, "w+");

    fwrite(X, sizeof(int), N*N, fp);

    fclose(fp);

  }

  int count(bool* currGrid, int height, int width, const std::string& prefix)
  {
    int counter = 0;
    std::cout << prefix << "Counting cells...." << std::endl;
    for (int y = 0; y < height; ++y){
      for (int x = 0; x < width; ++x){
        counter += currGrid[toLinearIndex(y, x, width)]; //Counting the cells;
      }
    }
    std::cout << prefix << "Number of alive cells: " << counter << std::endl;
    std::cout << prefix << "Living / Total Cells Percentage: "
      << (float)counter / (float)(width * height) << std::endl;

    return counter;
  }

  int countGhost(bool* currGrid, int height, int width, const std::string& prefix)
  {
    int counter = 0;
    std::cout << prefix << "Counting cells...." << std::endl;
    for (int y = 1; y < height+1; ++y){
      for (int x = 1; x < width+1; ++x){
        counter += currGrid[toLinearIndex(y, x, width + 2)]; //Counting the cells;
      }
    }
    std::cout << prefix << "Number of alive cells: " << counter << std::endl;
    std::cout << prefix << "Percent: " << (float)counter / (float)(width * height) << std::endl;

    return counter;
  }

  void print(bool* grid,size_t N )
  {
    std::cout<<std::endl;
    if ( N< 20){
      for ( int y =0 ; y<N;y++)
      {
        for (int x = 0;x<N;x++)
        {
          std::cout<<grid[toLinearIndex(y, x , N)];
        }
        std::cout<<std::endl;
      }
    }
    std::cout<<std::endl;
  }

} // namespace utilities
