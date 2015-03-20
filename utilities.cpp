#include "utilities.h"

void read_from_file(bool *X, std::string filename, size_t N){

  FILE *fp = fopen(filename.c_str(), "r+");

  int size = fread(X, sizeof(int), N*N, fp);

  std::cout << "elements: " <<  size << std::endl;

  fclose(fp);

}

void generate_table(bool *X, size_t N){
  srand(time(NULL));
  int counter = 0;

  for(size_t i=0; i<N; i++){
    for(size_t j=0; j<N; j++){
      X[ i * N + j] = ( (float)rand() / (float)RAND_MAX ) < THRESHOLD; 
      counter += X[i*N + j];
    }
  }

  std::cout << "Number of non zerow elements: " << counter << std::endl;
  std::cout << "Percent: " << (float)counter / (float)(N*N) << std::endl;
}


void save_table(bool *X, size_t N){

  FILE *fp;

  char filename[20];

  std::cout << filename << "table " << N << "x" << N << std::endl;

  std::cout <<"Saving table in file " << filename << std::endl;

  fp = fopen(filename, "w+");

  fwrite(X, sizeof(int), N*N, fp);

  fclose(fp);

}


