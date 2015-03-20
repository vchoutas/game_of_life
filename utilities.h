#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstdio>
#include <iostream>
#define THRESHOLD 0.3
#define GENERATIONS 10

void read_from_file(bool *X, std::string filename, size_t N);
void generate_table(bool *X, size_t N);
void save_table(int *X, int N);

#endif
