#include "game_of_life.h"

int main(int argc,char** argv)
{
  glutInit(&argc, argv);
  int N = 8;
  std::string fileName("table500x500.bin");
  GameOfLife *game = new GameOfLife(N);

  game->getNextGenerationWrapper(0); 
  glutMainLoop();
  return 0;
}

