#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H
#include "utilities.h"
#include "serial.h"


class GameOfLife
{
  public:
    explicit GameOfLife(std::string fileName);

    ~GameOfLife()
    {
      std::cout << "Destroying Game of Life Object!" << std::endl;
      delete[] currentGrid_;
      delete[] nextGrid_;
    }

    // The function that calculates the number of living neighbors cells.

    bool parseConfigFile(std::string fileName);

    void terminate(void);

    void play();
   
  private:
    static GameOfLife* ptr;
    int width_;
    int height_;
    bool *currentGrid_;
    bool *nextGrid_;

    std::string outputFileName_;
    std::string inputFileName_;
    int maxGenerationNumber_;
    int genCnt_;

    struct timeval startTime, endTime;

};

#endif // GAME_OF_LIFE_H
