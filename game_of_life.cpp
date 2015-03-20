#include "game_of_life.h"

const GLfloat GameOfLife::left = 0.0;
const GLfloat GameOfLife::right = 1.0;
const GLfloat GameOfLife::bottom = 0.0;
const GLfloat GameOfLife::top = 1.0;
const GLint GameOfLife::FPS = 25;
GLfloat GameOfLife::zoomFactor = 1;
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
    exit(-1);
  }
  nextGrid_ = new bool[width_ * height_];
  if ( nextGrid_ == NULL )
  {
    std::cout << "Could not allocate memory for the next generation Grid!" 
      << std::endl;
    exit(-1);
  }
  //bool BEACON_2[64] = {  0, 0, 0, 0, 0, 0, 0, 1,
                         //0, 0, 0, 0, 0, 0, 0, 0, 
                         //0, 0, 1, 1, 0, 0, 0, 0, 
                         //0, 0, 1, 0, 0, 0, 0, 0, 
                         //0, 0, 0, 0, 0, 1, 0, 0, 
                         //0, 0, 0, 0, 1, 1, 0, 0, 
                         //0, 0, 0, 0, 0, 0, 0, 0, 
                         //0, 0, 0, 0, 0, 0, 0, 1};
  for(size_t i = 0; i < N; i++)
  {
    for(size_t j = 0; j < N; j++)
    {
      currentGrid_[ i * N + j] = ( (float)rand() / (float)RAND_MAX ) 
        < THRESHOLD; 
      //currentGrid_[ i * N + j] = BEACON_2[ i * N + j];
    }
  }
}

void GameOfLife::initDisplay(void)
{
  glutInitWindowSize( windowWidth , windowHeight);
  glutInitWindowPosition(0, 0);
  glutCreateWindow("Game of Life");
  glClearColor(0, 0, 0, 0);

  glutDisplayFunc(GameOfLife::display);
  glutKeyboardFunc(GameOfLife::keyBoardCallBack);
}

void GameOfLife::reshape(int w , int h)
{
  windowWidth = w;
  windowHeight = h;

  glViewport(1000, 1000, windowWidth, windowHeight);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.f , windowWidth , windowHeight, 0.f ,0.f ,1.f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glutPostRedisplay();
}
void GameOfLife::display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  if (ptr == NULL)
  {
    std::cout << "Pointer not initialized ! " << std::endl;
    exit(-1);
  }
  glLoadIdentity();
  GLfloat xSize = zoomFactor * (right - left) / ptr->width_;
  GLfloat ySize = zoomFactor * (top - bottom) / ptr->height_;
  GLint width = ptr->width_;
  GLint height = ptr->height_;

  glBegin(GL_QUADS);
  for (GLint x = 0; x < width ; ++x) 
  {
    for (GLint y = 0; y < height; ++y) 
    {
      //ptr->currentGrid_[ y * width + x]  ?glColor3f(1,0,1):glColor3f(0,0,0);
      if ( !ptr->nextGrid_[y * width + x] && !ptr->currentGrid_[y*width + x])
        glColor3f(0,0,0);
      if ( ptr->nextGrid_[y * width + x] && !ptr->currentGrid_[y*width + x] )
        glColor3f(0,0.5,0);
      if (ptr->nextGrid_[y*width + x] && ptr->currentGrid_[y*width + x])
        glColor3f(1,0,0);
      if (!ptr->nextGrid_[y*width + x] && ptr->currentGrid_[y*width +x])
        glColor3f(0,0,1);
      glVertex2f( (x - width / 2 )* xSize , 
         - (y - height /2 ) * ySize);
      glVertex2f( (x + 1 - width / 2) * xSize , 
          -(y - height /2) * ySize );
      glVertex2f( (x + 1 - width / 2) * xSize , 
          -(y + 1- height /2) * ySize );
      glVertex2f( (x - width / 2)* xSize + left, 
          -(y + 1 - height / 2) * ySize );
    }
  }
  glEnd();

  glFlush();
  glutSwapBuffers();
}

void GameOfLife::keyBoardCallBack(unsigned char key, int x, int y)
{
  switch(key)
  {
    case '+':
      zoomFactor += 0.1;
      break;
    case '-':
      zoomFactor -= 0.1;
      break;
    default:
      break;
  }
  if (zoomFactor < 0)
    zoomFactor = 0.2;
}

void GameOfLife::getNextGenerationWrapper(int value)
{
  if (ptr == NULL)
  {
    std::cout << "The pointer to the function has not been initialized!" 
      << std::endl;
    exit(-1);
  }
  ptr->getNextGeneration(value);
  glutPostRedisplay();
  glutTimerFunc(1000 / ptr->FPS , GameOfLife::getNextGenerationWrapper, 0);
  //glutTimerFunc(2000 , GameOfLife::getNextGenerationWrapper, 0);

}


void GameOfLife::getNextGeneration(int value)
{
  for (size_t y = 0; y < height_; ++y)
  {
    size_t up = ( (y + height_ - 1) % height_) * width_;
    size_t center = y * width_;
    size_t down = ((y + 1) % height_) * width_;
    for (size_t x = 0; x < width_; ++x) 
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
  std::swap(currentGrid_, nextGrid_);
}

int GameOfLife::calcNeighbors(int x,int left,int right,int center,
    int up ,int down)
{
  
  return currentGrid_[left + up] + currentGrid_[x + up]
      + currentGrid_[right + up] + currentGrid_[left + center] 
      + currentGrid_[right + center] + currentGrid_[left + down] 
      + currentGrid_[x + down] + currentGrid_[right + down];
}
