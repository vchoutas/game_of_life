#include <iostream>
#include <cstdlib>
#include <string>

#include "utilities.h"

void computeIterationSerial(bool* m_data ,bool* m_resultData, size_t N) {
    for (size_t y = 0; y < N; ++y) {
    size_t y0 = ((y + N - 1) % N) * N;
    size_t y1 = y * N;
    size_t y2 = ((y + 1) % N) * N;
   
      for (size_t x = 0; x < N; ++x) {
      size_t x0 = (x + N - 1) % N;
      size_t x2 = (x + 1) % N;
  
    char aliveCells = m_data[x0 + y0] + m_data[x + y0] + m_data[x2 + y0]
      + m_data[x0 + y1] + m_data[x2 + y1]
      + m_data[x0 + y2] + m_data[x + y2] + m_data[x2 + y2];
  
    m_resultData[y1 + x] = aliveCells == 3 || (aliveCells == 2 && m_data[x + y1]) ? 1 : 0;
    }
  }
  std::swap(m_data, m_resultData);
 }

 int main (int argc, char** argv)
 {
   if ( argc < 2 )
   {
     std::cout << "Invalid number of arguments!" << std::endl;
     std::cout << "Usage : " << argv[0] << "N(size of grid)" <<
     "fileName(optional : file to read the grid)" << std::endl;
     return -1;
   }
   bool* grid;
   size_t N;
   N = atoi(argv[1]);
   grid = new bool[N*N];
   if (!grid)
   {
     std::cout << "Failed to allocate memory for the game of life" << 
      " grid !" << std::endl;
     return -1;
   }
   if (argc == 2)
   {
     generate_table(grid, N);
   }
   else
   {
     std::string fileName = argv[2];
     read_from_file(grid, fileName , N);
   }
   int generations = 2;
   bool* helpGrid = new bool[N*N];
   for (int i = 0 ; i < generations ; i++)
   {
    computeIterationSerial(grid,helpGrid,N);
   }
   
   return 0;
}
