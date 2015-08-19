#include <sys/time.h>
#include <fstream>
#include <sstream>

#include "game_of_life.h"

const GLfloat GameOfLife::left = 0.0;
const GLfloat GameOfLife::right = 1.0;
const GLfloat GameOfLife::bottom = 0.0;
const GLfloat GameOfLife::top = 1.0;
const GLint GameOfLife::FPS = 25;
GLfloat GameOfLife::zoomFactor = 1;
GLfloat GameOfLife::deltaX = 0.0f;
GLfloat GameOfLife::deltaY = 0.0f;
GLint GameOfLife::windowWidth  = 600;
GLint GameOfLife::windowHeight = 600 ;
GameOfLife* GameOfLife::ptr = NULL;

GameOfLife::GameOfLife(int N)
{
  // Seed the random number Generator
  srand(time(NULL));

  ptr = this;
  // Set the dimensions of the grid.
  width_ = N;
  height_ = N;

  initDisplay();

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

  colorArray_ = new color[width_ * height_];
  if (colorArray_ == NULL)
  {
    std::cout << "Could not allocate memory for the color Array!"
      << std::endl;
    std::exit(-1);
  }

  for(int i = 0; i < height_; i++)
  {
    for(int j = 0; j < width_; j++)
    {
      currentGrid_[i * width_ + j] = ( (float)rand() / (float)RAND_MAX )
        < THRESHOLD;
      colorArray_[i * width_ + j] = color(0, 0, 0);
      //currentGrid_[ i * N + j] = BEACON_2[ i * N + j];
    }
  }
}

GameOfLife::GameOfLife(std::string fileName): genCnt_(0)
{
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

  colorArray_ = new color[width_ * height_];
  if (colorArray_ == NULL)
  {
    std::cout << "Could not allocate memory for the color Array!"
      << std::endl;
    std::exit(-1);
  }

  // If the specified input file name is the "random" keyword
  // then create a random initial grid.
  if (inputFileName_.compare("random") == 0)
  {
    srand(time(NULL));
    for(int i = 0; i < height_; i++)
    {
      for(int j = 0; j < width_; j++)
      {
        currentGrid_[i * width_ + j] = ( (float)rand() / (float)RAND_MAX )
          < THRESHOLD;
      }
    }
  }
  else
    // Parse the grid from the file
    utilities::read_from_file(currentGrid_, inputFileName_, width_);

  std::cout << "Successfully created the initial grid!" << std::endl;

  if (displayFlag_)
    initDisplay();

  for(int i = 0; i < height_; i++)
  {
    for(int j = 0; j < width_; j++)
    {
      colorArray_[i * width_ + j] = color(0, 0, 0);
    }
  }
  ptr = this;

  std::cout << "Created Game of Life Object!" << std::endl;
  return;
}

/**
 * @brief Initialize all the functions used for displaying the grid.
 */
void GameOfLife::initDisplay(void)
{
  glutInitWindowSize(windowWidth , windowHeight);
  glutInitWindowPosition(0, 0);
  windowId_ = glutCreateWindow("Game of Life");
  glClearColor(0, 0, 0, 0);

  glutDisplayFunc(GameOfLife::display);
  glutIdleFunc(GameOfLife::getNextGenerationWrapper);
  glutKeyboardFunc(GameOfLife::keyBoardCallBack);
  glutSpecialFunc(GameOfLife::arrowKeyCallback);
}

void GameOfLife::reshape(int w , int h)
{
  windowWidth = w;
  windowHeight = h;

  glViewport(0, 0, windowWidth, windowHeight);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.f, windowWidth , windowHeight, 0.f, 0.f , 1.f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glutPostRedisplay();
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
      // Parse the option specifying whether to display the game or not.
      else if (command.compare("display") == 0)
      {
        std::string displayStr;
        ss >> displayStr;
        if (ss.fail())
        {
          std::cout << "Could not read the value for the display flag, setting it to false!" << std::endl;
          displayFlag_ = false;
        }
        else
          displayFlag_ = displayStr.compare("true") == 0;
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

  std::cout << "Finished Reading the configuration file!" << std::endl;
  return true;
}

void GameOfLife::play()
{
  std::cout << "Starting to play!" << std::endl;

  if (!displayFlag_)
  {
    gettimeofday(&startTime, NULL);
    for (genCnt_ = 0; genCnt_ < maxGenerationNumber_; ++genCnt_)
      getNextGeneration();
    gettimeofday(&endTime, NULL);
    double execTime = (double)((endTime.tv_usec - startTime.tv_usec)
        /1.0e6 + endTime.tv_sec - startTime.tv_sec);
    std::cout << "Execution Time is = " << execTime << std::endl;
  }
  else
  {
    gettimeofday(&startTime, NULL);
    glutMainLoop();
  }

  std::cout << "Finished playing the game of Life!" << std::endl;
}

void GameOfLife::display()
{
  // Clear the buffer.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Calculate the size of each cell in each direction.
  GLfloat xSize = zoomFactor * (right - left) / ptr->width_;
  GLfloat ySize = zoomFactor * (top - bottom) / ptr->height_;
  GLint width = ptr->width_;
  GLint height = ptr->height_;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glTranslatef(deltaX, deltaY, 0.0f);

  glMatrixMode(GL_MODELVIEW);
  // Load the identity transformation matrix.
  glLoadIdentity();
  // Define the scale transformation so as to properly view the grid.
  glScalef(xSize, ySize, 1.0f);
  // Apply a translation transformation so as to place the center of the grid
  // on the center of the window and move it when the user moves it using the
  // keyboard arrow keys.
  glTranslatef(-width / 2.0f, height / 2.0f, 0.0f);

  glBegin(GL_QUADS);

  // TO DO : Replace vertex drawing with faster method(vertex array or VBO) for
  // faster rendering.
  for (GLint y = 0; y < height; ++y)
  {
    for (GLint x = 0; x < width ; ++x)
    {
      int index = y * width + x;
      // At this point, the nextGrid array contains the information abou the last generation
      // of the game.
      // If the current cell was dead and is revived
      if (ptr->currentGrid_[index] && !ptr->nextGrid_[index])
        ptr->colorArray_[index] = color(0, 128, 0);

      // If the cell was alive and died.
      if (!ptr->currentGrid_[index] && ptr->nextGrid_[index])
        ptr->colorArray_[index] = color(128, 0, 0);
      else
        ptr->colorArray_[index].red > 0 ? ptr->colorArray_[index].red-- : 0;

      // If the cell remains alive.
      if (ptr->currentGrid_[index])
        ptr->colorArray_[index].green >= 255 ? ptr->colorArray_[index].green = 255:
          ptr->colorArray_[index].green++;

      // Update the current color.
      glColor3ub(ptr->colorArray_[index].red, ptr->colorArray_[index].green,
          ptr->colorArray_[index].blue);
      // Draw the vertex.
      glVertex2f(x, -y - 1);
      glVertex2f(x + 1, -y - 1);
      glVertex2f(x + 1, -y);
      glVertex2f(x, -y);
    }
  }
  glEnd();
  glFlush();
  glutSwapBuffers();

}

void GameOfLife::keyBoardCallBack(unsigned char key, int x, int y)
{
  // TO DO : Add Arrow Key support
  switch(key)
  {
    case '+':
      zoomFactor += 0.1f;
      break;
    case 'r':
    case 'R':
      zoomFactor = 1.0f;
      deltaX = 0.0f;
      deltaY = 0.0f;
      break;
    case '-':
      zoomFactor -= 0.1f;
      break;
    // If the Escape key was pressed then free the allocated resources and std::exit.
    case char(27):
      ptr->terminate();
      break;
    default:
      break;
  }
  if (zoomFactor < 0)
    zoomFactor = 0.2;
  glutPostRedisplay();
}

void GameOfLife::arrowKeyCallback(int key, int x, int y)
{
  switch (key)
  {
    case GLUT_KEY_LEFT: //left
      deltaX -= 0.005f;
      break;
    case GLUT_KEY_UP: //up
      deltaY += 0.005f;
      break;
    case GLUT_KEY_RIGHT: //right
      deltaX += 0.005f;
      break;
    case GLUT_KEY_DOWN: //down
      deltaY -= 0.005f;
      break;
    default:
      break;
  }
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
  glutDestroyWindow(windowId_);
  std::exit(0);
  return;
}

void GameOfLife::getNextGenerationWrapper()
{
  if (ptr == NULL)
  {
    std::cout << "The pointer to the function has not been initialized!"
      << std::endl;
    std::exit(-1);
  }
  if (ptr->genCnt_ > ptr->maxGenerationNumber_)
    ptr->terminate();

  // gettimeofday(&ptr->startTime, NULL);
  ptr->getNextGeneration();
  // gettimeofday(&ptr->endTime, NULL);

  // double nextGenTime = (double)((ptr->endTime.tv_usec - ptr->startTime.tv_usec)
      // /1.0e6 + ptr->endTime.tv_sec - ptr->startTime.tv_sec);

  // std::cout << std::endl << "Next Gen Time = " << nextGenTime << std::endl;

  glutPostRedisplay();
  return;
}


void GameOfLife::getNextGeneration()
{
  genCnt_++;
  for (int y = 0; y < height_; ++y)
  {
    size_t up = ( (y + height_ - 1) % height_) * width_;
    size_t center = y * width_;
    size_t down = ((y + 1) % height_) * width_;
    for (int x = 0; x < width_; ++x)
    {
      size_t left = (x + width_ - 1) % width_;
      size_t right = (x + 1) % width_;

      int livingNeighbors = calcNeighbors(x , left, right,center,up,down);
      nextGrid_[center + x] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && currentGrid_[x + center]) ? 1 : 0;
    }
  }
  // Set the next generation grid as the current on for the next iteration
  // of the algorithm.
  // TO DO : Make into MACRO
  std::swap(currentGrid_, nextGrid_);
  return;
}

int GameOfLife::calcNeighbors(int x, int left, int right, int center,
    int up, int down)
{
  return currentGrid_[left + up] + currentGrid_[x + up]
      + currentGrid_[right + up] + currentGrid_[left + center]
      + currentGrid_[right + center] + currentGrid_[left + down]
      + currentGrid_[x + down] + currentGrid_[right + down];
}
