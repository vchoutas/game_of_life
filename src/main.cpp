#include "game_of_life.h"

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
  GameOfLife game(argv[1]);

  game.play();

  return 0;
}

