#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include "utilities.h"
#include "serial.h"
#ifdef __APPLE__
#  include <OpenGL/gl.h>
#  include <OpenGL/glu.h>
#  include <GLUT/glut.h>
#else

#  include <GL/gl.h>
#  include <GL/glu.h>
#  include <GL/glut.h>
#endif


class GameOfLife
{
  public:
    // Default Constructor.
    GameOfLife(): left(-1.0f), right(1.0f), top(1.0f), bottom(-1.0f)
    {}
    // Constructor that creates a random square grid
    explicit GameOfLife(int N);

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

    // Function to initialize the necessary OpenGL components.
    void initDisplay(void);
    static void display(void);
    static void reshape(int w , int h);

    static void arrowKeyCallback(int key, int x, int y);
    static void keyBoardCallBack(unsigned char key, int x, int y);
    // The function that gets the next generation of the game.
    static void getNextGenerationWrapper();

    void updateColors(int x, int y);
    void initColorArray(void);

    void initTexture(void);
    
  private:
    static GameOfLife* ptr;
    int width_;
    int height_;
    int windowId_;
    bool *currentGrid_;
    bool *nextGrid_;
    GLubyte* colorArray_;

    std::string outputFileName_;
    std::string inputFileName_;
    bool displayFlag_;
    int maxGenerationNumber_;
    GLuint gl_pixelBufferObject;
    GLuint gl_texturePtr;
    int genCnt_;

    struct timeval startTime, endTime;

    static GLfloat zoomFactor;
    static GLfloat deltaX;
    static GLfloat deltaY;
    static GLint windowWidth;
    static GLint windowHeight;
    const GLfloat left ;
    const GLfloat right ;
    const GLfloat bottom;
    const GLfloat top;
};

#endif // GAME_OF_LIFE_H
