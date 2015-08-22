#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>


void read_from_file(bool *X, std::string filename, size_t N){

  FILE *fp = fopen(filename.c_str(), "r+");

  int size = fread(X, sizeof(int), N*N, fp);

  printf("elements: %d\n", size);

  fclose(fp);

}


