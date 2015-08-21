#include "game_of_life.h"
#include "simple_cuda_gol.cuh"

int main(int argc,char* argv[])
{
  if (argc < 2)
  {
    std::cout << "Usage : " << argv[0] << std::endl
      << "\t configFileName The name of the configuration File"
      << std::endl;
    return -1;
  }
  glutInit(&argc, argv);

  SimpleCudaGoL simpleCudaGame(argv[1]);
  simpleCudaGame.play();

  return 0;
}
