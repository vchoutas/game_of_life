#ifndef SIMPLE_CUDA_GOL_H
#define SIMPLE_CUDA_GOL_H

#include "utilities.h"


#include <cuda.h>
#include <cuda_runtime.h>

__global__ void simpleNextGenerationKernel(bool* currentGrid, bool* nextGrid, const int N);
__device__ inline int calcNeighborsKernel(bool* currentGrid, int x, int left, int right, int center, int up , int down);

class SimpleCudaGoL
{
  public:
    // Default Constructor.
    SimpleCudaGoL(){}
    // Constructor that creates a random square grid
    explicit SimpleCudaGoL(int N);

    explicit SimpleCudaGoL(std::string fileName);

    ~SimpleCudaGoL()
    {
      std::cout << "Destroying Simple Cuda Game of Life Object!" << std::endl;
      delete[] currentGrid_;
      delete[] nextGrid_;
    }

    bool parseConfigFile(std::string fileName);

    void terminate(void);

    void play();

    // Function to initialize the necessary OpenGL components.
    void initDisplay(void);
    static void display(void);
    static void reshape(int w , int h);

    static void arrowKeyCallback(int key, int x, int y);
    static void keyBoardCallBack(unsigned char key, int x, int y);
    // The function that gets the next generation of the game.
    static void getNextGenerationWrapper();
    void getNextGeneration();
    inline int calcNeighbors(int x, int left, int right, int center, int up , int down);

  private:
    static SimpleCudaGoL* ptr;
    int width_;
    int height_;
    int windowId_;
    bool *currentGrid_;
    bool *nextGrid_;
    color* colorArray_;

    std::string outputFileName_;
    std::string inputFileName_;
    bool displayFlag_;
    int maxGenerationNumber_;
    int genCnt_;

    static GLfloat zoomFactor;
    static GLfloat deltaX;
    static GLfloat deltaY;
    static GLint windowWidth;
    static GLint windowHeight;
    static const GLfloat left ;
    static const GLfloat right ;
    static const GLfloat bottom;
    static const GLfloat top;
    static const GLint FPS;
};

#endif // SIMPLE_CUDA_GOL_H
