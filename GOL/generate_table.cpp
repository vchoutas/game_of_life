#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THRESHOLD 0.4

void generate_table(bool *X, size_t N){
  srand(time(NULL));
  int counter = 0;
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      X[i*N + j] = ( (float)rand() / (float)RAND_MAX ) < THRESHOLD;
      counter += X[i*N + j];
    }
  }


  printf("Number of non zerow elements: %d\n", counter);
  printf("Percent: %f\n", (float)counter / (float)(N*N));
}

void generate_still_table(bool *X, size_t N){

   bool Y[64] = {  0, 0, 0, 0, 0, 0, 1, 1,
                   0, 0, 1, 1, 0, 0, 0, 0, 
                   0, 1, 0, 0, 1, 0, 0, 0, 
                   0, 0, 1, 1, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 1, 1};


   bool A[64] = {  1, 0, 0, 0, 0, 0, 0, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   1, 0, 0, 0, 0, 0, 0, 1};

   bool C[64] = {  1, 0, 0, 0, 0, 0, 0, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 1, 1, 1, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   1, 0, 0, 0, 0, 0, 0, 1};


  
   bool D[64] = {  0, 0, 0, 0, 0, 0, 0, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 1, 1, 0, 0, 0, 0, 
                   0, 0, 1, 1, 0, 0, 0, 0, 
                   0, 0, 0, 0, 1, 1, 0, 0, 
                   0, 0, 0, 0, 1, 1, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 1};

   bool E[64] = {  0, 0, 0, 0, 0, 0, 0, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 1, 1, 0, 0, 0, 0, 
                   0, 0, 1, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 1, 0, 0, 
                   0, 0, 0, 0, 1, 1, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 1};

                   
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      X[i*N + j] = D[i*N + j];
    }
  }
}
//~ generate_period_table(bool *X, size_t N){
//~ 
//~ X = [ ]
//~ 
  //~ }

//~ 
  //~ printf("Number of non zerow elements: %d\n", counter);
  //~ printf("Percent: %f\n", (float)counter / (float)(N*N));
//~ }

void save_table(int *X, int N){

  FILE *fp;

  char filename[20];

  sprintf(filename, "table%dx%d.bin", N, N);

  printf("Saving table in file %s\n", filename);

  fp = fopen(filename, "w+");

  fwrite(X, sizeof(int), N*N, fp);

  fclose(fp);

}


