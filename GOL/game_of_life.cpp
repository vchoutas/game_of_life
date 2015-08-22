#include <sys/time.h>
#include <fstream>
#include <sstream>
#include "game_of_life.h"
//#include "simple_cuda.cuh"

GameOfLife::GameOfLife(std::string fileName){
  bool parseFlag = parseConfigFile(fileName);
  if (!parseFlag)
  {
    std::cout << "Game of Life will not begin!Exiting now!" << std::endl;
    std::exit(-1);
  }

  currentGrid_ = new bool[width_ * height_];
  if ( currentGrid_ == NULL )
  {
    std::cout << "Could not allocate memory for the current Grid!" <<
      std::endl;
    std::exit(-1);
  }
  nextGrid_ = new bool[width_ * height_];
  if ( nextGrid_ == NULL )
  {
    std::cout << "Could not allocate memory for the next generation Grid!"
      << std::endl;
    std::exit(-1);
  }

  // If the specified input file name is the "random" keyword
  // then create a random initial grid.
  if (inputFileName_.compare("random") == 0){
    utilities::generate_table(currentGrid_,height_);//It will create a square matrix only
  }
  else{
    // Parse the grid from the file
    utilities::read_from_file(currentGrid_, inputFileName_, width_);
    utilities::count(currentGrid_,height_,width_);
  }
  std::cout << "Successfully created the initial grid!" << std::endl;

}



void GameOfLife::play(){
  std::cout << "Starting to play!" << std::endl;
  gettimeofday(&startTime, NULL);
    for (int i = 0; i < maxGenerationNumber_; ++i){
      serial::getNextGeneration(currentGrid_,nextGrid_,height_,width_);
      std::swap(currentGrid_,nextGrid_);
    }
    gettimeofday(&endTime, NULL);
    double execTime = (double)((endTime.tv_usec - startTime.tv_usec)
        /1.0e6 + endTime.tv_sec - startTime.tv_sec);

  std::cout << "Execution Time is = " << execTime << std::endl;
  utilities::count(currentGrid_,height_,width_);
  std::cout << "Finished playing the game of Life!" << std::endl;
}


bool GameOfLife::parseConfigFile(std::string fileName)
{
  std::cout << "Parsing Input File!" << std::endl;

  std::ifstream configFile;
  configFile.open(fileName.c_str());
  if (!configFile.is_open())
  {
    std::cout << "Could not open the configuration file!" << std::endl;
    return false;
  }

  std::string line;
  std::getline(configFile, line);
  while (configFile)
  {
    // Ignore tab lines, carriage return, newline and the # characters
    if ((line.find_first_not_of(" \t\r\n") != std::string::npos) && (line[0] != '#'))
    {
      std::stringstream ss(line);
      std::string command;
      ss >> command;

      // Get the size of the Game Of Life Grid.
      if (command.compare("width") == 0)
      {
        ss >> width_;
        if (ss.fail())
        {
          std::cout << "Could not read the width of the grid!" << std::endl;
          return false;
        }
      }
      else if (command.compare("height") == 0)
      {
        ss >> height_;
        if (ss.fail())
        {
          std::cout << "Could not read the height of the grid!" << std::endl;
          return false;
        }
      }
      // Get the name of the file where the board is stored.
      else if (command.compare("boardFileName") == 0)
      {
        ss >> inputFileName_;
        if (ss.fail())
        {
          std::cout << "Could not read the name of the file containing the board!" << std::endl;
          return false;
        }
      }
      // Get the maximum number of generations for the game.
      else if (command.compare("generationNumber") == 0)
      {
        ss >> maxGenerationNumber_;
        if (ss.fail())
        {
          std::cout << "Could not parse the max number of generations!" << std::endl;
          return false;
        }
      }
      // If present, read the name of the file where the
      else if (command.compare("outputFile") == 0)
      {
        ss >> outputFileName_;
        if (ss.fail())
          std::cout << "Could not read the name of the output grid file!"
            << " The result will not be saved!"<< std::endl;
      }
    } // End of If clause for invalid characters.

    // Get the next line of the file.
    getline(configFile, line);
  } // End of While loop.
  return true;
}


void GameOfLife::terminate()
{
  gettimeofday(&endTime, NULL);
  double execTime = (double)((endTime.tv_usec - startTime.tv_usec)
      /1.0e6 + endTime.tv_sec - startTime.tv_sec);
  std::cout << "Execution Time is = " << execTime << std::endl;
  std::cout << "Terminating Game of Life!" << std::endl;
  delete[] currentGrid_;
  delete[] nextGrid_;
  return;
}
