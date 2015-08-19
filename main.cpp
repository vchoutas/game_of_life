#include "game_of_life.h"

int main(int argc,char* argv[])
{
  // if (argc < 3)
  // {
    // std::cout << "Usage : " << argv[0] << std::endl
      // << "\t inputTableFile(if 'random' a random grid will be created)"
      // << std::endl
      // << "\t N (size of grid)" << std::endl
      // << "\t toggleDisplay : true if yes, false if not(default is false)"
      // << std::endl;
    // return -1;
  // }
  glutInit(&argc, argv);
  int N = atoi(argv[1]);
  std::cout << N << std::endl;
  std::string fileName("table500x500.bin");
  GameOfLife game(N);

  game.getNextGenerationWrapper(0);
  glutMainLoop();

  return 0;
}

