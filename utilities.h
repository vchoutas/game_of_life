#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstdio>
#include <iostream>
#ifdef __APPLE__
#  include <OpenGL/gl.h>
#else
#  include <GL/gl.h>
#endif
#define THRESHOLD 0.3
#define GENERATIONS 10

void read_from_file(bool *X, std::string filename, size_t N);
void generate_table(bool *X, size_t N);
void save_table(int *X, int N);

struct color
{
  public:
    GLubyte red;
    GLubyte green;
    GLubyte blue;

    color()
    {
      red = 0;
      green = 0;
      blue = 0;
    }
    color(GLubyte r, GLubyte g, GLubyte b)
    {
      red = r;
      green = g;
      blue = b;
    }
};
#endif
