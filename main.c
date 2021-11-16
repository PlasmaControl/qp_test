#include "qp_solver.h"
#include <stdio.h>

int main(void) {
  const float lambda=0.1;
  const float phi[VEPN]={0,0};
  const float Gamma[VEPN][VEPN]={{1,0},{0,1}};
  float xOut[VEPN]={2,.3};
  const size_t N = VEPN;
  const float l[VEPN]={-1,-1};
  const float u[VEPN]={1,1};

  vep_box(lambda, N, phi, Gamma, xOut, l, u);
  printf("FINAL RESULTS FROM main.c:\n");
  for (int i=0; i<N; i++) {
    printf("%f\n",xOut[i]);
  }

  return 0;
}
