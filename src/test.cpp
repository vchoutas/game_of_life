#include "serial.h"
#include  "test.h"

int test1(){

  bool* GRID;
  bool* NGRID;
  int pass=1;
  GRID = new bool[N*N];
  NGRID = new bool[N*N];
  copyAtoB(STILL,GRID);
  for (int g = 0; g<500;g++){
  
    serial::getNextGeneration(GRID, NGRID,N,N);
    if (!isEqual(NGRID,STILL)){
       pass = 0;
     }
     
     std::swap(GRID, NGRID);
   }
   delete GRID;
   delete NGRID;
   return pass;
 }

int test2(){

  bool* GRID;
  bool* NGRID;
  int pass=1;
  GRID = new bool[N*N];
  NGRID = new bool[N*N];
  copyAtoB(BLINKER_1,GRID);
  for (int g = 0; g<100;g++){
  
    serial::getNextGeneration(GRID, NGRID,N,N);
     if (!isEqual(NGRID,BLINKER_1)&&(g%2)){
       pass = 0;
     }
    if(!isEqual(NGRID,BLINKER_2)&&!(g%2)){
       pass = 0;
     }
     
     std::swap(GRID, NGRID);
  }
  
   delete GRID;
   delete NGRID;
   return pass;
}
int test3(){

  bool* GRID;
  bool* NGRID;
  int pass=1;
  GRID = new bool[N*N];
  NGRID = new bool[N*N];
  copyAtoB(BEACON_1,GRID);
  for (int g = 0; g<100;g++){
  
    serial::getNextGeneration(GRID, NGRID,N,N);
     if (!isEqual(NGRID,BEACON_1)&&(g%2)){
       pass = 0;
     }
    if(!isEqual(NGRID,BEACON_2)&&!(g%2)){
       pass = 0;
     }
      std::swap(GRID, NGRID);
  }
  delete GRID;
  delete NGRID;
  return pass;
}

int main(int argc,char** argv)
{
 const char *result[2];
 result[0] = "failed";
 result[1] = "passed";
 std::cout<<"STILL test "<<result[test1()]<<std::endl;
 std::cout<<"BEACON test "<<result[test2()]<<std::endl;
 std::cout<<"BLINKER test "<<result[test3()]<<std::endl;
 return 0;
}



void copyAtoB (bool* A , bool *B)
{
  for(int i=0; i<N*N; i++){
        B[i]=A[i];
      }
  }
bool isEqual(bool* A,bool* B)
{
  for (int i =0 ;i<N*N;i++){if (!A[i]==B[i]){return false;}}
  return true;
}
void print (bool* grid )
   {
     int counter=0;
      if ( N< 20){
       for ( int y =0 ; y<N;y++)
       {
         for (int x = 0;x<N;x++)
         {
           counter +=grid[y*N + x];
           std::cout<<grid[y*N + x];
         }
         std::cout<<std::endl;
       }
     }
     std::cout<<"Number of alive cells"<<counter<<std::endl;
   }

