#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include "utilities.h"

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
    GameOfLife(){}
    // Constructor that creates a random square grid
    GameOfLife(int N);

    GameOfLife(int N, std::string fileName){};

    // Constructor that creates a random orthogonal grid of size MxN
    GameOfLife(int M, int N){}

    // Constructor that reads from a file an orthogonal grid of size MxN
    GameOfLife(int M,int N, std::string fileName){}
    
    // The function that calculates the number of living neighbors cells.
    int calcNeighbors(int x,int left,int right,int center,int up ,int down);
  
    // Function to initialize the necessary OpenGL components.
    void initDisplay(void);
    static void display(void);
    static void reshape(int w , int h);
    static void keyBoardCallBack(unsigned char key, int x, int y);
    // The function that gets the next generation of the game.
    static void getNextGenerationWrapper(int value);
    void getNextGeneration(int value);
  private:
    static GameOfLife* ptr;
    int width_;
    int height_;
    bool *currentGrid_;
    bool *nextGrid_;

    static GLfloat zoomFactor;
    static GLint windowWidth;
    static GLint windowHeight;
    static const GLfloat left ;
    static const GLfloat right ;
    static const GLfloat bottom;
    static const GLfloat top;
    static const GLint FPS;
};

#endif // GAME_OF_LIFE_H
