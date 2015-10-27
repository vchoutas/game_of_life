#include "game_of_life.cuh"

int main(int argc, char *argv[])
{
  GameOfLife game(argc, argv);
  game.play();

  return 0;
}
