# Conway's Game of Life
This repository contains an implementation of Conway's Game of Life in C++ and CUDA Code by @[Vassilis Choutas](https://github.com/vasilish) and @[Konstantinos Chamzas](https://github.com/ChamzasKonstantinos).
The project was the 3rd Assignment of the **Parallel and Distributed Systems** course of the Electrical
and Computer Engineering Department of the Aristotle University of Thessaloniki.

## Game Description

The universe of the Game of Life is an infinite two-dimensional orthogonal grid of square cells, each of which is in one of two possible states, alive or dead. Every cell interacts with its eight neighbours, which are the cells that are horizontally, vertically, or diagonally adjacent.

The rules of the game are simple:
 1. Any live cell with fewer than two live neighbours dies, as if caused by under-population.
 2. Any live cell with two or three live neighbours lives on to the next generation.
 3. Any live cell with more than three live neighbours dies, as if by over-population.
 4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

In order to simulate the infinite grid we implemented a cyclic world version.

<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/5918727/13548956/cc257308-e304-11e5-886b-096fa8a2bb0f.gif" alt="Gosper's Glider Gun"
 width=300 height=300/>
</p>

For more information and other interesting facts visit [Conway's Game of Life on Wikipedia](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)

## CUDA Kernels

### Simple Cuda Kernel
In the first kernel each cell is assigned one cell, fetches it's neighbors and decides according to the rules above
whether it is alive or not in the next generation.

### Grid Sized Loop Kernel
The second kernel uses a 2D version of a grid sized loop to calculate each generation. The number of cells
that corresponds to each thread is a input parameter specified in the configuration file by the *cellsPerThread* entry.

### Shared Memory Kernel
The final (and fastest) kernel makes use of the on-chip shared memory. When the kernel is launched, all threads in a block copy their corresponding cells to the shared memory array, thus limiting redundant global memory loads, and then calculate the status of each cell in the next generation.

## Display
The project has OpenGL support for viewing the game's evolution. We made use of the CUDA-OpenGL interoperability
functionality to update the texture array when running the CUDA version.
The GLUT and OpenGL libraries are required for the execution of the game.

### Colors
- When the game starts, all cells that are initially alive are colored green.
- Then, if a cell dies it is colored red and gradually vanishes from the board.
- As long as a cell is alive it's blue channel componet is increased.

An Example:

<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/5918727/13549668/416decea-e313-11e5-95fb-2a1c6ff521bf.gif"
  alt="Game of Life"
 width=300 height=300/>
</p>

## Building

### Requirements

The following libraries are required to build the code:

  1. CUDA (obviously :P)
  2. CMake >= 2.8
  3. FreeGLUT library

### Compiling the code

We made use of **CMake** in order to build the project.
The process is pretty simple:

 1. *mkdir build # Create a directory where the compilation will be performed*
 2. *cd build*
 3. *cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc # This is done for systems where the user may use another compiler that may not be compatible with NVIDIA's nvcc)*
 4. *make*

Please make sure you have correctly set up all CUDA environment variables. The above were tested
on Ubuntu 14.04 and compiled with gcc 4.8.4 .

## TO DO
 - Finish Documentation
 - Improve color update kernel. It is a major bottleneck. Check memory access
 - Add more input options, such as parsing files RLE files.
