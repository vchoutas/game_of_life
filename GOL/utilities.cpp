#include "utilities.h"

namespace utilities
{

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
        counter += X[i*N + j];
      }
    }

    std::cout << "Number of non zerow elements: " << counter << std::endl;
    std::cout << "Percent: " << (float)counter / (float)(N*N) << std::endl;
  }
  void generate_ghost_table(bool *Grid,bool *GhostGrid, size_t N){


   for(size_t i=0; i<N+2; i++){
      for(size_t j=0; j<N+2; j++){
        GhostGrid[ i * N + j] = 0;
      }
    }
    for(size_t i=0; i<N; i++){
      for(size_t j=0; j<N; j++){
        GhostGrid[ (i+1) * (N+2) + j+1] = Grid[i*N + j];
      }
    }

    std::cout<<"Ghost grid  created"<<std::endl;
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
        counter += currGrid[y * width + x]; //Counting the cells;
      }
    }
    std::cout << prefix << "Number of alive cells: " << counter << std::endl;
    std::cout << prefix << "Living / Total Cells Percentage: "
      << (float)counter / (float)(width * height) << std::endl;

    return counter;
  }

  int countGhost(bool* currGrid, int height, int width)
  {
    int counter = 0;
    std::cout << "Counting cells...." << std::endl;
    for (int y = 1; y < height+1; ++y){
      for (int x = 1; x < width+1; ++x){
        counter += currGrid[y * (width+2) + x]; //Counting the cells;
      }
    }
    std::cout << "Number of alive cells: " << counter << std::endl;
    std::cout << "Percent: " << (float)counter / (float)(width * height) << std::endl;

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
           std::cout<<grid[y*N + x];
         }
         std::cout<<std::endl;
       }
     }
     std::cout<<std::endl;
   }

} // namespace utilities
