#include <sys/time.h>
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
  int counter = 0;
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

  colorArray = new color[width_ * height_];
  if (colorArray == NULL)
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
      colorArray[i * width_ + j] = color(0, 0, 0);
      //currentGrid_[ i * N + j] = BEACON_2[ i * N + j];
    }
  }
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
  glOrtho(0.f , windowWidth , windowHeight, 0.f, 0.f , 1.f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glutPostRedisplay();
}
void GameOfLife::display()
{
  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);

  // Clear the buffer.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (ptr == NULL)
  {
    std::cout << "Pointer not initialized ! " << std::endl;
    exit(-1);
  }

  // Calculate the size of each cell in each direction.
  GLfloat xSize = zoomFactor * (right - left) / ptr->width_;
  GLfloat ySize = zoomFactor * (top - bottom) / ptr->height_;
  GLint width = ptr->width_;
  GLint height = ptr->height_;

  // Load the identity transformation matrix.
  glLoadIdentity();
  // Define the scale transformation so as to properly view the grid.
  glScalef(xSize, ySize, 1.0f);
  // Apply a translation transformation so as to place the center of the grid
  // on the center of the window and move it when the user moves it using the
  // keyboard arrow keys.
  glTranslatef(-width / 2.0f + deltaX, height / 2.0f + deltaY, 0.0f);

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
        ptr->colorArray[index] = color(0, 128, 0);

      // If the cell was alive and died.
      if (!ptr->currentGrid_[index] && ptr->nextGrid_[index])
        ptr->colorArray[index] = color(128, 0, 0);
      else
        ptr->colorArray[index].red > 0 ? ptr->colorArray[index].red-- : 0;

      // If the cell remains alive.
      if (ptr->currentGrid_[index])
        ptr->colorArray[index].green >= 255 ? ptr->colorArray[index].green = 255:
          ptr->colorArray[index].green++;

      // Update the current color.
      glColor3ub(ptr->colorArray[index].red, ptr->colorArray[index].green,
          ptr->colorArray[index].blue);
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

  gettimeofday(&endTime, NULL);

  double renderingTime = (double)((endTime.tv_usec - startTime.tv_usec)
      /1.0e6 + endTime.tv_sec - startTime.tv_sec);
  // std::cout << "Rendering Time = " << renderingTime << std::endl;
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
    // If the Escape key was pressed then free the allocated resources and exit.
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
      deltaX -= 1.0f;
      break;
    case GLUT_KEY_UP: //up
      deltaY += 1.0f;
      break;
    case GLUT_KEY_RIGHT: //right
      deltaX += 1.0f;
      break;
    case GLUT_KEY_DOWN: //down
      deltaY -= 1.0f;
      break;
    default:
      break;
  }
}


void GameOfLife::terminate()
{
  std::cout << "Terminating Game of Life!" << std::endl;
  delete[] currentGrid_;
  delete[] nextGrid_;
  glutDestroyWindow(windowId_);
  exit(0);
  return;
}

void GameOfLife::getNextGenerationWrapper(int value)
{
  if (ptr == NULL)
  {
    std::cout << "The pointer to the function has not been initialized!"
      << std::endl;
    exit(-1);
  }
  struct timeval startTime, endTime;

  gettimeofday(&startTime, NULL);
  ptr->getNextGeneration(value);
  gettimeofday(&endTime, NULL);

  double nextGenTime = (double)((endTime.tv_usec - startTime.tv_usec)
      /1.0e6 + endTime.tv_sec - startTime.tv_sec);

  gettimeofday(&startTime, NULL);
  glutPostRedisplay();
  gettimeofday(&endTime, NULL);

  // std::cout << std::endl << "Next Gen Time = " << nextGenTime << std::endl;

  // TO DO : Add max generation limit
  glutTimerFunc(1000 / ptr->FPS , GameOfLife::getNextGenerationWrapper, 0);
}


void GameOfLife::getNextGeneration(int value)
{
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
