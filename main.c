#include "qp_solver.h"
#include <stdio.h>

int main(void) {
  float lambda=0;
  // very weird: "if length is a variable, you need memset, 
  // but if length is a compile-time constant, then the 
  // statement compiles just fine." (StackOverflow)
#define VEPN 2
  size_t N = VEPN;
  float phi[VEPN]={0};
  float Gamma[VEPN][VEPN]={{1,0},{0,1}};
  float xOut[VEPN]={2,.3};
  float l[VEPN]={-1,-1};
  float u[VEPN]={1,1};

  vep_box(lambda, N, phi, Gamma, xOut, l, u);
  printf("FINAL RESULTS for x^2 (-1,1) 2D case");
  printf("Calculated  |    True\n");
  for (int i=0; i<N; i++) {
    printf("%f    |    %f\n",xOut[i],0.0);
  }

  float phi_1[VEPN]={-1,-1};
  float xOut_1[VEPN]={3,3};

  vep_box(lambda, N, phi_1, Gamma, xOut_1, l, u);
  printf("\n");
  printf("FINAL RESULTS for x^2 - |x| (-1,1) 2D case");
  printf("Calculated  |    True\n");
  for (int i=0; i<N; i++) {
    printf("%f    |    %f\n",xOut_1[i],1.0);
  }

  
  vep_box(lambda, N, phi_1, Gamma, xOut_1, l, u);
  printf("\n");
  printf("FINAL RESULTS for x^2 - |x| (-1,1) 2D case");
  printf("Calculated  |    True\n");
  for (int i=0; i<N; i++) {
    printf("%f    |    %f\n",xOut_1[i],1.0);
  }


  return 0;
}
