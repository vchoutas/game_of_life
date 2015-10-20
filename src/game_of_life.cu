#include <algorithm>
#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <unistd.h>

#include "game_of_life.cuh"
#include "cuda_kernels.cuh"

GLfloat GameOfLife::zoomFactor = 1;
GLfloat GameOfLife::deltaX = 0.0f;
GLfloat GameOfLife::deltaY = 0.0f;
GLint GameOfLife::windowWidth  = 800;
GLint GameOfLife::windowHeight = 800 ;
GameOfLife* GameOfLife::ptr = NULL;

/**
 * @brief Constructor that takes as input the path to the
 * configuration file for the Game of Life.
 * @param fileName[const std::string&] The path to the
 * configuration file.
 */
GameOfLife::GameOfLife(const std::string& fileName):
  left(-1.0f), right(1.0f), top(1.0f), bottom(-1.0f), gpuOn_(true),
  cellPerThread_(4)
{
  bool parseFlag = parseConfigFile(fileName);
  if (!parseFlag)
  {
    std::cout << "Game of Life will not begin!Exiting now!" << std::endl;
    std::exit(-1);
  }

  ptr = this;
  // Calculate the size of the padded array.
  ghostCellNum_ = width_ + 2;

  currentGrid_ = new bool[ghostCellNum_ * ghostCellNum_];
  if ( currentGrid_ == NULL )
  {
    std::cout << "Could not allocate memory for the current Grid!" <<
      std::endl;
    std::exit(-1);
  }
  nextGrid_ = new bool[ghostCellNum_ * ghostCellNum_];
  if ( nextGrid_ == NULL )
  {
    std::cout << "Could not allocate memory for the next generation Grid!"
      << std::endl;
    std::exit(-1);
  }

  // Create the initial array and pad it with the ghost cells.
  createGhostArray();
  std::cout << "Successfully created the initial grid!" << std::endl;

  if (!setupCuda())
  {
    std::cout << "Failed to initialize CUDA related data. Exiting!" << std::endl;
    std::exit(-1);
  }

  // If display is enabled :
  if (displayFlag_)
  {
    // Set up all the necessary OpenGL functions.
    bool openglSetupFlag = setupOpenGL();
    if (!openglSetupFlag)
    {
      std::cout << "Could not initialize Graphics!" << std::endl;
      std::exit(-1);
    }
  }

  std::cout << "Created Game of Life Object!" << std::endl;
  return;
}

bool GameOfLife::setupCuda()
{
  cudaMalloc((void**) &currentGridDevice_, ghostCellNum_ * ghostCellNum_);
  cudaCheckErrors("Current Generation Grid Device memory Allocation Error!");

  cudaMalloc((void**) &nextGridDevice_, ghostCellNum_ * ghostCellNum_);
  cudaCheckErrors("Next Generation Grid Device memory Allocation Error!");

  if (currentGridDevice_ == NULL)
  {
    std::cout << "Could not allocate Device Memory for the current Grid." << std::endl;
    return false;
  }
  if (nextGridDevice_ == NULL)
  {
    std::cout << "Could not allocate Device Memory for the next generation Grid." << std::endl;
    return false;
  }

  // Copy the starting grid from the main memory to the GPU memory.
  cudaMemcpy(currentGridDevice_, currentGrid_,
      ghostCellNum_ * ghostCellNum_ * sizeof(bool), cudaMemcpyHostToDevice);
  cudaCheckErrors("Error when copying the initial grid to the GPU");

  cudaMemcpy(nextGridDevice_, nextGrid_,
      ghostCellNum_ * ghostCellNum_ * sizeof(bool), cudaMemcpyHostToDevice);
  cudaCheckErrors("Error when copying the next generation's grid to the GPU");

  return true;
}

bool GameOfLife::createGhostArray()
{
  bool* inputArray = new bool[width_ * height_];
  if (inputArray == NULL)
  {
    std::cout << "Could not allocate memory for the initial temporary array." << std::endl;
    return false;
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
        inputArray[i * width_ + j] = ( (float)rand() / (float)RAND_MAX )
          < THRESHOLD;
      }
    }
  }
  else
    // Parse the grid from the file
    utilities::read_from_file(inputArray, inputFileName_, width_);

  for (int i = 0; i < ghostCellNum_; ++i) {
    for (int j = 0; j < ghostCellNum_; ++j) {
        currentGrid_[i * ghostCellNum_ + j] = 0;
        nextGrid_[i * ghostCellNum_ + j] = 0;
    }
  }

  // Copy the game board.
  for (int i = 1; i < ghostCellNum_ - 1; ++i) {
    for (int j = 1; j < ghostCellNum_ - 1; ++j) {
        currentGrid_[i * ghostCellNum_ + j] = inputArray[(i - 1) * width_ + j - 1];
    }
  }

  delete[] inputArray;

  return true;
}


/**
 * @brief Initialize all the functions used for displaying the grid.
 */
bool GameOfLife::setupOpenGL(void)
{
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(windowWidth , windowHeight);
  glutInitWindowPosition(0, 0);
  windowId_ = glutCreateWindow("Game of Life");

  glewInit();

  /* if (!glewIsSupported("GL_VERSION_2_0")) */
  /* { */
    /* std::cerr << "ERROR: Support for necessary OpenGL extensions missing." << std::endl; */
    /* std::exit(-1); */
  /* } */

  glutReportErrors();

  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  std::cout << "Creating Texture!" << std::endl;
  bool textureSetupFlag = textureSetup();
  if (!textureSetupFlag)
  {
    std::cout << "Failed to initialize the textures necessary for rendering the Game of Life. Exiting!" << std::endl;
    return false;
  }
  std::cout << "Finished creating texture object!" << std::endl;

  glutDisplayFunc(GameOfLife::displayCallback);
  glutIdleFunc(GameOfLife::gameLoopCallback);
  glutKeyboardFunc(GameOfLife::keyBoardCallBack);
  glutSpecialFunc(GameOfLife::arrowKeyCallback);

  return true;
}

/**
 * @brief Initializes the color array and binds it to an OpenGL texture.
 */
bool GameOfLife::textureSetup(void)
{
  if (colorBufferId_)
  {
    glDeleteBuffers(1, &colorBufferId_);
    colorBufferId_ = 0;
  }

  // Allocate memory for the color array as a 3 channel image with size equal to
  // width_ times height_.
  colorArray_ = new GLubyte[3 * width_ * height_];

  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &gl_texturePtr);
  glBindTexture(GL_TEXTURE_2D, gl_texturePtr);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width_, height_, 0, GL_RGB,
      GL_UNSIGNED_BYTE, NULL);

  glBindTexture(GL_TEXTURE_2D, 0);

  glGenBuffers(1, &colorBufferId_);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, colorBufferId_);

  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width_ * height_ * 3 * sizeof(GLubyte), NULL, GL_STREAM_COPY);

  cudaGraphicsGLRegisterBuffer(&cudaPboResource_,
        colorBufferId_, cudaGraphicsMapFlagsWriteDiscard);
  cudaCheckErrors("Could Not Register the CUDA OpenGL buffer");

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  bool colorCreationResult = createColorArrays();
  if (!colorCreationResult)
  {
    std::cout << "Could not initialize the color arrays!" << std::endl;
    return false;
  }

  return true;
}

void GameOfLife::reshape(int w, int h)
{
  windowWidth = w;
  windowHeight = h;

  glViewport(0, 0, windowWidth, windowHeight);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.f, windowWidth , windowHeight, 0.f, -1.f , 1.f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glutPostRedisplay();
}


bool GameOfLife::parseConfigFile(const std::string& fileName)
{
  std::cout << "Parsing configuration file!" << std::endl;

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
      else if (command.compare("gpuEnabled") == 0)
      {
        std::string gpuFlag;
        ss >> gpuFlag;
        if (ss.fail())
          std::cout << "Could not read the value of the GPU enable flag!"
            << "Using default value: !" << gpuOn_ << std::endl;

        std::transform(gpuFlag.begin(), gpuFlag.end(), gpuFlag.begin(), ::toupper);
        if (gpuFlag.compare("TRUE") == 0)
          gpuOn_ = true;
        else
          gpuOn_ = false;
      }
      else if (command.compare("cellsPerThread") == 0)
      {
        ss >> cellPerThread_;
        if (ss.fail())
        {
          std::cout << "Failed to read the value of the number of cells to be processed" <<
            " by each thread. Using the default value = 2!"<< std::endl;
        }
      }
    } // End of If clause for invalid characters.

    // Get the next line of the file.
    getline(configFile, line);
  } // End of While loop.

  std::cout << "Finished Reading the configuration file!" << std::endl;
  return true;
}

void GameOfLife::play(void)
{
  std::cout << "Starting to play!" << std::endl;


  if (!displayFlag_)
  {
    gettimeofday(&startTime, NULL);
    for (int i = 0; i < maxGenerationNumber_; ++i){
      getNextGeneration(currentGrid_,nextGrid_,height_,width_);
      std::cout << "Yo" << std::endl;
      genCnt_++;
      std::swap(currentGrid_,nextGrid_);
    }
    gettimeofday(&endTime, NULL);
    double execTime = (double)((endTime.tv_usec - startTime.tv_usec)
        /1.0e6 + endTime.tv_sec - startTime.tv_sec);
    std::cout << "Execution Time is = " << execTime << std::endl;
  }
  else
  {
    genCnt_ = 0;
    gettimeofday(&startTime, NULL);
    frameCounter_ = 0;
    fps_ = 0;

    glutMainLoop();
  }

  std::cout << "Finished playing the game of Life!" << std::endl;
}

bool GameOfLife::createColorArrays(void)
{

  // Bind the PBO object that will be used to update the texture image.
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, colorBufferId_);
  // Get a pointer to the array.
  colorArray_ = (GLubyte* )glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_READ_WRITE);

  // Initialize it to black.
  for(int i = 0; i < height_; i++)
  {
    for(int j = 0; j < width_; j++)
    {
      colorArray_[i * width_ * 3 + j * 3]  = 0;
      // Color all living cells green.
      if (currentGrid_[i * width_ + j])
        colorArray_[i * width_ * 3 + j * 3 + 1]  = 255;
      else
        colorArray_[i * width_ * 3 + j * 3 + 1]  = 0;
      colorArray_[i * width_ * 3 + j * 3 + 2]  = 0;
    }
  }

  // Unmap the pointer used to access the texture array
  // in order to initialize it.
  glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);

  // Allocate memory for the CUDA color array.
  cudaMalloc((void**) &colorArrayDevice_, 3 * width_ * height_ * sizeof(GLubyte));
  cudaCheckErrors("Color Array Device memory Allocation Error!");

  if (colorArrayDevice_ == NULL)
  {
    std::cout << "Failed to allocate memory for the color array on the GPU." << std::endl;
    return false;
  }

  // Initialize the CUDA color array by copying the corresponding host
  // array.
  cudaMemcpy(ptr->colorArrayDevice_, ptr->colorArray_,
      3 * ptr->width_ * ptr->height_ * sizeof(GLubyte), cudaMemcpyHostToDevice);
  cudaCheckErrors("Error when copying the color array from the Host to the Device");

  return true;
}

void GameOfLife::updateColors(int x, int y)
{
  int index = (y + 1) * ghostCellNum_ + x + 1;

  int colorIndex = 3 * (y * width_ + x);
  // If the current cell was dead and is revived
  if (nextGrid_[index] && !currentGrid_[index])
  {
    colorArray_[colorIndex]  = 0;
    colorArray_[colorIndex + 1]  = 128;
    // colorArray_[colorIndex + 2]  = 0;
  }
  // If the cell was alive and died.
  if (!nextGrid_[index] && currentGrid_[index])
  {
    colorArray_[colorIndex] = 200;
    colorArray_[colorIndex + 1] = 0;
    // colorArray_[colorIndex + 2] = 0;
  }
  else
    colorArray_[colorIndex] > 0 ? colorArray_[colorIndex]-- : colorArray_[colorIndex] = 0;

  // If the cell remains alive.
  if (nextGrid_[index])
    // colorArray_[colorIndex + 1] >= 255 ? colorArray_[colorIndex + 1] = 255:
      // colorArray_[colorIndex + 1]++;
    colorArray_[colorIndex + 2] >= 255 ? colorArray_[colorIndex + 2] = 255:
      colorArray_[colorIndex + 2]++;

  return;
}

void GameOfLife::displayCallback()
{
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  // Calculate the size of each cell in each direction.
  GLfloat xSize = zoomFactor * (ptr->right - ptr->left) / ptr->width_;
  GLfloat ySize = zoomFactor * (ptr->top - ptr->bottom) / ptr->height_;

  GLint width = ptr->width_;
  GLint height = ptr->height_;


  glMatrixMode(GL_TEXTURE);
  // glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glTranslatef(deltaX, deltaY, 0.0f);
  // glScalef(xSize, ySize, 1.0f);

  glMatrixMode(GL_MODELVIEW);

  // Load the identity transformation matrix.
  glLoadIdentity();

  // Define the scale transformation so as to properly view the grid.
  glScalef(xSize, ySize, 1.0f);
  // Apply a translation transformation so as to place the center of the grid
  // on the center of the window and move it when the user moves it using the
  // keyboard arrow keys.
  glTranslatef(-width / 2.0f, height / 2.0f, 0.0f);

  glBindTexture(GL_TEXTURE_2D, ptr->gl_texturePtr);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, ptr->colorBufferId_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);

  glBegin(GL_QUADS);

  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(0.0f, 0.0f);

  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(width, 0.0f);

  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(width, -height);

  glTexCoord2f(0.0f, 1.0f);
  glVertex2f( 0.0f, -height);

  glEnd();

  // if (ptr->gpuOn_)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  ptr->drawGameInfo();

  glFlush();
  glutSwapBuffers();

  return;
}

void GameOfLife::drawGameInfo()
{
  // Update the number of frames.
  frameCounter_++;
  // Calculate the elapsed time.
  gettimeofday(&endTime, NULL);
  double elapsedTime = (double)((endTime.tv_usec - startTime.tv_usec)
        /1.0e6 + endTime.tv_sec - startTime.tv_sec);

  // If a second has passed since the last call.
  if (elapsedTime > 1.0)
  {
    // Calculate the number of frames per second.
    fps_ = frameCounter_ / elapsedTime;
    // gettimeofday(&startTime, NULL);
    startTime = endTime;
    frameCounter_ = 0;
  }

  glColor3f(1.0f, 1.0f, 1.0f);
  glRasterPos2f(0, - 1.0f * height_ / 8.0f);

  std::string activeDeviceType("Active Device: ");
  if (gpuOn_)
    activeDeviceType += std::string("GPU");
  else
    activeDeviceType += std::string("CPU");

  for (int i = 0; i < activeDeviceType.size(); ++i) {
    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, (int)activeDeviceType[i]);
  }

  // Draw the frames per second on the screen.
  std::stringstream ss;
  ss << "FPS: " << fps_;

  std::string fpsString(ss.str());

  glColor3f(1.0f, 1.0f, 1.0f);
  glRasterPos2f(0, - 3.0f * height_ / 16.0f);
  for (int i = 0; i < fpsString.size(); ++i) {
    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, (int)fpsString[i]);
  }

  glPopMatrix();

  glColor3f(1.0f, 1.0f, 1.0f);
  return;
}

void GameOfLife::keyBoardCallBack(unsigned char key, int x, int y)
{
  // TO DO : Add Arrow Key support
  switch(key)
  {
    case '+':
      zoomFactor += 0.01f;
      break;
    case 'r':
    case 'R':
      zoomFactor = 1.0f;
      deltaX = 0.0f;
      deltaY = 0.0f;
      break;
    case '-':
      zoomFactor -= 0.01f;
      break;
    case 't':
    case 'T':
      ptr->gpuOn_ ^= 1;
      if (ptr->gpuOn_)
      {
        std::cout << "Running Game of Life on GPU" << std::endl;
        // Copy all the necessary data for the game of life from the
        // CPU to the GPU device.

        // Copy the starting grid from the main memory to the GPU memory.
        cudaMemcpy(ptr->currentGridDevice_, ptr->currentGrid_,
            ptr->ghostCellNum_ * ptr->ghostCellNum_ * sizeof(bool), cudaMemcpyHostToDevice);
        cudaCheckErrors("Error when copying the current grid from the Host to the Device");

        // Copy the next generation grid from the main memory to the GPU memory.
        cudaMemcpy(ptr->nextGridDevice_, ptr->nextGridDevice_,
            ptr->ghostCellNum_ * ptr->ghostCellNum_ * sizeof(bool), cudaMemcpyHostToDevice);
        cudaCheckErrors("Error when copying the next grid from the Host to the Device");

        // Copy the array containing the color for each cell
        // from the main memory to the GPU memory.
        cudaMemcpy(ptr->colorArrayDevice_, ptr->colorArray_,
            3 * ptr->width_ * ptr->height_ * sizeof(bool), cudaMemcpyHostToDevice);
        cudaCheckErrors("Error when copying the color array from the Host to the Device");
      }
      else
      {
        std::cout << "Running Game of Life on CPU" << std::endl;
        // Copy all the data from the GPU back to the host's memory.

        // Copy the starting grid from the GPU to the main memory.
        cudaMemcpy(ptr->currentGrid_, ptr->currentGridDevice_,
            ptr->ghostCellNum_ * ptr->ghostCellNum_ * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Error when copying the current grid from the Device to the Host");

        // Copy the next generation grid from the GPU to the main memory.
        cudaMemcpy(ptr->nextGridDevice_, ptr->nextGridDevice_,
            ptr->ghostCellNum_ * ptr->ghostCellNum_ * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Error when copying the next grid from the Device to the Host");

        // Copy the array containing the color for each cell
        // from the GPU to the main memory.
        cudaMemcpy(ptr->colorArray_, ptr->colorArrayDevice_,
            3 * ptr->width_ * ptr->height_ * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Error when copying the color array from the Device to the Host");
      }
      break;
    case '1':
      ptr->gpuMethod_ = 1;
      break;
    case '2':
      ptr->gpuMethod_ = 2;
      break;
    case '3':
      ptr->gpuMethod_ = 3;
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
    case GLUT_KEY_RIGHT: //right
      deltaX += 0.005f;
      break;
    case GLUT_KEY_UP: //up
      deltaY += 0.005f;
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

  cudaDeviceReset();
  glutDestroyWindow(windowId_);
  std::exit(0);
  return;
}

void GameOfLife::gameLoopCallback()
{
  if (ptr == NULL)
  {
    std::cout << "The pointer to the function has not been initialized!"
      << std::endl;
    std::exit(-1);
  }
  if (ptr->genCnt_ == 0)
  {
    gettimeofday(&ptr->startTime, NULL);
  }

  if (ptr->genCnt_ > ptr->maxGenerationNumber_)
    ptr->terminate();

  if (!ptr->gpuOn_)
  {
    ptr->getNextGeneration(ptr->currentGrid_, ptr->nextGrid_, ptr->height_, ptr->width_);
    std::swap(ptr->currentGrid_, ptr->nextGrid_);
  }
  else
  {
    ptr->getNextGenDevice(ptr->currentGridDevice_, ptr->nextGridDevice_, ptr->height_, ptr->width_);
    std::swap(ptr->currentGridDevice_, ptr->nextGridDevice_);
  }
  ptr->genCnt_++;

  glutPostRedisplay();
  return;
}

void GameOfLife::updateGhostCells()
{
  // Initialize the top and bottom ghost rows.
  for (int j = 1; j < ghostCellNum_ - 1; ++j) {
    // Copy the bottom row to the top ghost row.
    currentGrid_[j] = currentGrid_[(ghostCellNum_ - 2) * ghostCellNum_ + j];
    // Copy the top row to the bottom ghost row.
    currentGrid_[(ghostCellNum_ - 1) * ghostCellNum_ + j] = currentGrid_[ghostCellNum_ + j];
  }

  for (int i = 1; i < ghostCellNum_ - 1; ++i) {
    // Copy the right most column to the first column of the padded array.
    currentGrid_[i * ghostCellNum_] = currentGrid_[i * ghostCellNum_ + ghostCellNum_ - 2];
    // Copy the left most column to the last column of the padded array.
    currentGrid_[i * ghostCellNum_ + ghostCellNum_ - 1] = currentGrid_[i * ghostCellNum_ + 1];
  }

  // Copy the corners of the game board.
  currentGrid_[0] = currentGrid_[(ghostCellNum_ - 2) * ghostCellNum_ + ghostCellNum_ - 2];
  currentGrid_[ghostCellNum_ - 1] = currentGrid_[(ghostCellNum_ - 2) * ghostCellNum_ + 1];
  currentGrid_[(ghostCellNum_ - 1) * ghostCellNum_] = currentGrid_[ghostCellNum_ - 2];
  currentGrid_[(ghostCellNum_ - 1) * ghostCellNum_ + ghostCellNum_ - 1] = currentGrid_[ghostCellNum_ + 1];

  return;
}

/**
 * @brief The serial CPU version of the Game of Life
 */
void GameOfLife::getNextGeneration(bool* currGrid, bool* nextGrid, int height, int width)
{
  updateGhostCells();

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, colorBufferId_);
  colorArray_ = (GLubyte* )glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_READ_WRITE);
  for (int y = 1; y < ghostCellNum_ - 1; ++y)
  {
    size_t up = (y - 1) * ghostCellNum_;
    size_t center = y * ghostCellNum_;
    size_t down = (y + 1) * ghostCellNum_;
    for (int x = 1; x < ghostCellNum_ - 1; ++x)
    {

      size_t left = x- 1;
      size_t right = x + 1;

      int livingNeighbors = cuda_kernels::calcNeighbors(currGrid ,x, left, right,
          center, up, down);
      nextGrid[center + x] = livingNeighbors == 3 ||
        (livingNeighbors == 2 && currGrid[x + center]) ? 1 : 0;

      updateColors(x - 1, y - 1);
    }
  }

  glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
  // glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  return;
}

void GameOfLife::getNextGenDevice(bool* currentGridDevice, bool* nextGridDevice, int height, int width)
{
  dim3 threadNum(16, 16);
  dim3 blocks;
  dim3 ghostMatThreads(16, 1);
  dim3 ghostGridRowsSize(width_ / ghostMatThreads.x + 1, 1);
  dim3 ghostGridColSize(height_ / ghostMatThreads.x + 1, 1);

  cudaGraphicsMapResources(1, &cudaPboResource_, 0);
  cudaStreamSynchronize(0);

  size_t numBytes;
  cudaGraphicsResourceGetMappedPointer((void**)&colorArrayDevice_, &numBytes, cudaPboResource_);

  switch(gpuMethod_)
  {
    default:
    case 1:
      // Copy the Contents of the current and the next grid
      blocks = dim3(width_ / threadNum.x + 1, height_ / threadNum.y + 1);

      cuda_kernels::updateGhostRows<<< ghostGridRowsSize, ghostMatThreads>>>(currentGridDevice, ghostCellNum_,
          ghostCellNum_);
      cuda_kernels::updateGhostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice,
          ghostCellNum_, ghostCellNum_);
      cuda_kernels::updateGhostCorners<<< 1, 1 >>>(currentGridDevice, ghostCellNum_,
          ghostCellNum_);
      cuda_kernels::simpleGhostNextGenerationKernel<<<blocks, threadNum>>>(currentGridDevice, nextGridDevice,
          width_, colorArrayDevice_);
      break;

    case 2:
      blocks = dim3(width_ / (threadNum.x * cellPerThread_) + 1,
          height_ / (threadNum.y * cellPerThread_) + 1);

      cuda_kernels::updateGhostRows<<< ghostGridRowsSize, ghostMatThreads>>>(currentGridDevice, ghostCellNum_,
          ghostCellNum_);
      cuda_kernels::updateGhostCols<<< ghostGridColSize, ghostMatThreads>>>(currentGridDevice,
          ghostCellNum_, ghostCellNum_);
      cuda_kernels::updateGhostCorners<<< 1, 1 >>>(currentGridDevice, ghostCellNum_, ghostCellNum_);
      cuda_kernels::multiCellGhostGridLoop<<<blocks, threadNum>>>(currentGridDevice,
          nextGridDevice, width_, colorArrayDevice_);

      break;
    case 3:
      std::cout << "Multiple Cells on Shared Memory" << std::endl;
      break;
  }

  cudaGraphicsUnmapResources(1, &cudaPboResource_, 0);
  cudaStreamSynchronize(0);

  return;
}


