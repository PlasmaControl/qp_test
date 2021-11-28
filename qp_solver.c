#include "qp_solver.h"
#include "nstx_math.h"
#include <string.h>
#include <stdbool.h>
#include <float.h> // for FLT_MAX
#include <stdio.h> //for debugging (printf's)
// A1.3.1

// 0 is no output
// 1 shows iterations and parameters
// 2 shows a ton of stuff along the way
#ifndef VERBOSE_LEVEL
#define VERBOSE_LEVEL 1
#endif

static float clamp(float const min, float const max, float val) {
  if(val < min) val = min;
  if(val > max) val = max;
  return val;
}

// Note C99 allows you to use "N" from the previous argument (kinda weird), but in general 
// C forces you to specify the length for every dimension greater than 1
void vep_box(float const lambda, size_t const N, float const phi[N], float const Gamma[N][N], float xOut[N], float const l[N], float const u[N]) {
  /* Dan's original code, unclear from spec why these are the bounds but prob particular to his problem
     float u[N];
     float l[N];
     for (size_t i = 0; i < N; ++i) {
     u[i] = 1.0f - xOut[i];
     l[i] = -xOut[i];
     }
  */

  // A1.2.1
  float xBest[N];
  float xInit[N];
  for (size_t i = 0; i < N; ++i) {
    //xInit[i] = clamp(l[i], u[i], 0.0f);
    // change it so xOut is the initial guess, rather than 0
    xInit[i] = clamp(l[i], u[i], xOut[i]);
    xBest[i] = xInit[i];
  }

#if VERBOSE_LEVEL>=1
  printf("-----INITIALIZATION-----\n");
  for (int i=0; i<N; i++) {
    printf("\txBest[i]: %f\n",xBest[i]);
  }
#endif

  float cost(size_t const N, float const x[N]) {
    float t[N];
    memset(t, 0, sizeof(t));
    nstx_matrixMult1d2d(N, N, x, Gamma, t);

    float c = 0.0f;
    for (size_t i = 0; i < N; ++i)
      c += 0.5f * t[i] * x[i] + x[i] * phi[i];
    return c;
  }
  float fInit = cost(N, xInit);
  float fBest = fInit;

  static struct Scalars {
    size_t NI;
    size_t NK;
    float cb;
  } scalars = { 100, 4, 0.1f };
  for (size_t iNewton = 0; iNewton < scalars.NI; ++iNewton) {
    // A1.3.2
    // g is \grad{f} for f the cost function
    float g[N];
    memset(g, 0, sizeof(g));
    nstx_matrixMult1d2d(N, N, xInit, Gamma, g);

    for (size_t i = 0; i < N; ++i)
      g[i] += phi[i];
#if VERBOSE_LEVEL>=2
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
	printf("Gamma[i][j]: %f\n",Gamma[i][j]);
      }
      printf("xInit[i]: %f\n",xInit[i]);
      printf("g[i]: %f\n",g[i]);
    }
#endif

    // A1.4.1
    // binding is true if x is currently outside the range and planning to head further out
    // during the gradient descent
    bool binding[N];
    for (size_t i = 0; i < N; ++i)
      binding[i] = ( xInit[i] <= l[i] && g[i] > 0.0f ) || ( xInit[i] >= u[i] && g[i] < 0.0f );

    // A1.4.2
    // Hessian of cost function (i.e. Gamma since it's quadratic), except offdiag is 0 
    // if we don't want to move in the corresponding direction due to constraints
    float hRed[N][N];
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
	if (i == j)
	  hRed[i][j] = Gamma[i][j];
	else
	  hRed[i][j] = binding[i] || binding[j]? 0.0f : Gamma[i][j];

    float hRedInv[N][N];
    nstx_matrixInvert(N, hRed, hRedInv);

#if VERBOSE_LEVEL>=2
    printf("\n");
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
	printf("hRed[i][j]: %f\n",hRed[i][j]);
      }
    }
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
	printf("hRedInv[i][j]: %f\n",hRedInv[i][j]);
      }
    }
#endif

    // d is the search direction
    // f(x+d) ~ f(x) + d^T \nabla{f} + 1/2 d^T hRed d
    // so to optimize the \delta f, set d = -hRed^-1 \nabla f
    // this is "Newton's method"
    float d[N];
    memset(d, 0, sizeof(d));
    nstx_matrixMult2d1d(N, N, hRedInv, g, d);
    for (size_t i = 0; i < N; ++i)
      d[i] *= -1.0f;

#if VERBOSE_LEVEL>=2
    printf("\n");
    for (int i=0; i<N; ++i) {
      printf("Initial d[i]: %f\n",d[i]);
    }
    printf("\n");
#endif

    float fLow = FLT_MAX;
    float xLow[N];
    // this is the "line search" method to sample
    // a bunch of points along the Newton direction
    // with different step sizes
    // but we clamp to ensure we stay inbounds
    // note the step size gets exponentially smaller
    for (size_t k = 0; k < scalars.NK; ++k) {
      float xTest[N];
      for (size_t i = 0; i < N; ++i) {
	float const uk = u[i] - xInit[i];
	float const lk = l[i] - xInit[i];
	float const dk = clamp(lk, uk, d[i]);
	xTest[i] = xInit[i] + dk;
	d[i] *= scalars.cb;
      }

      float fTest = cost(N, xTest);
      if (fTest < fLow) {
	fLow = fTest;
	for (size_t i = 0; i < N; ++i)
	  xLow[i] = xTest[i];
      }
    }
    fInit = fLow;

    if (fLow < fBest) {
      for (size_t i = 0; i < N; ++i)
	// xBest = xInit + lambda*(xLow-xInit)
	xBest[i] = (1-lambda)*xInit[i] + lambda*xLow[i];
    }
    fBest = cost(N, xBest);
    
    for (size_t i = 0; i < N; ++i)
      xInit[i] = xBest[i];
    

#if VERBOSE_LEVEL>=1
    printf("-----ITERATION %d RESULT-----\n",iNewton+1);
    for (int i=0; i<N; i++) {
      printf("\txBest[i]: %f\n",xBest[i]);
    }    
#endif
  }

#if VERBOSE_LEVEL>=1
  printf("\n");
  printf("lambda (for scaling of final stepsize in iteration): %f\n",lambda);
  printf("cb (for line search exponential steps): %f\n",scalars.cb);
  for (int i=0; i<N; i++)
    printf("l[i]: %f\n",l[i]);
  for (int i=0; i<N; i++)
    printf("u[i]: %f\n",u[i]);
#endif
  for (size_t i = 0; i < N; ++i) 
    // original code was not simply outputting xBest, I think
    // it was incorrect
    xOut[i] = xBest[i]; //+= lambda * xBest[i];
}
