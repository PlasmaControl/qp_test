#include "qp_solver.h"
#include "nstx_math.h"
#include <stdio.h>
#include <stddef.h> //for size_t

void qp_solve(size_t const N, size_t const M,
	      float const G[N][N], float const P[N][N], float const A[M][N], 
	      float const rho, float const sigma, float const alpha, 
	      float const q[N], 
	      float const l[M], float const u[M], 
	      float const x0[N], 
	      float const eps, size_t const maxiter,
	      float xOut[N],
	      float yOut[M],
	      float r[2]) {
  // nstx_matrixMult2d1d(M, N, A, x0, xOut);
  // xOut[0]=1;
}
