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
#  include <OpenGL/glu.h>
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  include <GL/freeglut.h>
#  include <GL/gl.h>
#  include <GL/glu.h>
#  include <GL/glut.h>
#endif

#define THRESHOLD 0.3
#define GENERATIONS 10

#define SWAP(x, y) do {typeof(x) SWAP = x; x = y; y = SWAP; } while(0)

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
          msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

namespace utilities
{
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
} // namespace utilities

typedef utilities::color color;

#endif
