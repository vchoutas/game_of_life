#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include "utilities.h"

#ifdef __APPLE__
#  include <OpenGL/gl.h>
#  include <OpenGL/glu.h>
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  include <GL/freeglut.h>
#  include <GL/gl.h>
#  include <GL/glu.h>
#  include <GL/glut.h>
#endif


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iomanip>


/**
 * @class GameOfLife
 * @brief Class used to set up and play the Game of Life.
 */
class GameOfLife
{
  public:
    // Default Constructor.
    GameOfLife(): left(-1.0f), right(1.0f), top(1.0f), bottom(-1.0f)
    {}
    // Constructor that creates a random square grid
    explicit GameOfLife(int N);

    explicit GameOfLife(const std::string& fileName);

    GameOfLife(int argc, char** argv);

    ~GameOfLife()
    {
      std::cout << "Destroying Game of Life Object!" << std::endl;
      releaseAllocatedMemory();
    }

    void initGame(const std::string& fileName);

    void releaseAllocatedMemory()
    {
      delete[] currentGrid_;
      delete[] nextGrid_;
      delete[] colorArray_;
      cudaFree(currentGridDevice_);
      cudaFree(nextGridDevice_);
      cudaFree(colorArrayDevice_);
    }

    bool createGhostArray();

    // The function that calculates the number of living neighbors cells.

    bool parseConfigFile(const std::string& fileName);

    void terminate(void);

    void play();

    void printHelpMessage(const std::string& executableName);

    void printInstructions();

    static void displayCallback(void);
    static void reshape(int w, int h);

    static void arrowKeyCallback(int key, int x, int y);
    static void keyBoardCallBack(unsigned char key, int x, int y);
    // The function that gets the next generation of the game.
    static void gameLoopCallback();

    int calcNeighbors(bool* currGrid,int x, int left, int right, int center,
        int up, int down);
    void getNextGeneration(bool* currGrid,bool* nextGrid,int height,int width);

    void getNextGenDevice();

    void updateGhostCells();
    void updateColors(int x, int y);

    bool createColorArrays(void);

    bool textureSetup(void);

    void drawGameInfo();

    // Function to initialize the necessary OpenGL components.
    bool setupOpenGL(void);

    bool setupCuda();

  private:
    static GameOfLife* ptr;
    // Function used to decide if to calculate and print the execution
    // time for the game.
    bool calcExecTime_;
    /// The width of the grid.
    int width_;
    /// The height of the grid.
    int height_;
    int windowId_;
    bool* currentGrid_;
    bool* nextGrid_;
    GLubyte* colorArray_;

    /// Integer variable used to decide which kernel to execute.
    int gpuMethod_;

    int ghostCellNum_;
    /// Boolean variable
    bool gpuOn_;

    // GPU memory pointers.
    bool* currentGridDevice_;
    bool* nextGridDevice_;
    GLubyte* colorArrayDevice_;

    GLuint colorBufferId_;
    cudaGraphicsResource* cudaPboResource_;

    /// Name of the file where the final Game of Life board
    /// will be written.
    std::string outputFileName_;
    std::string inputFileName_;
    /// Flag for deciding whether to display the progression
    /// of the game or not.
    bool displayFlag_;
    /// The maximum number of generations that the game will be run.
    int maxGenerationNumber_;
    GLuint gl_pixelBufferObject;
    GLuint gl_texturePtr;
    /// The current number of generations of the game.
    int genCnt_;

    int cellPerThread_;

    int frameCounter_;
    double avgDrawTime_;
    double timeCounter_;

    struct timeval startTime, endTime;

    static GLfloat zoomFactor;
    static GLfloat deltaX;
    static GLfloat deltaY;
    GLint windowWidth_;
    GLint windowHeight_;

    const GLfloat left ;
    const GLfloat right ;
    const GLfloat bottom;
    const GLfloat top;
};

#endif // GAME_OF_LIFE_H
