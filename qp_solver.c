#include "qp_solver.h"
#include "nstx_math.h"
#include <string.h>
#include <stdbool.h>
#include <float.h> // for FLT_MAX
#include <stdio.h> //for testing
// A1.3.1
static float clamp(float const min, float const max, float val) {
  if(val < min) val = min;
  if(val > max) val = max;
  return val;
}

void vep_box(float const lambda, size_t const N, float const phi[VEPN], float const Gamma[VEPN][VEPN], float xOut[VEPN], float const l[VEPN], float const u[VEPN]) {
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
    xInit[i] = clamp(l[i], u[i], 0.0f);
    xBest[i] = xInit[i];
  }

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
  } scalars = { 1, 4, 0.1f };
  for (size_t iNewton = 0; iNewton < scalars.NI; ++iNewton) {
    // A1.3.2
    float g[N];
    memset(g, 0, sizeof(g));
    nstx_matrixMult1d2d(N, N, xInit, Gamma, g);

    for (int i=0; i<N; i++) {
      printf("%f\n",g[i]);
    }

    for (size_t i = 0; i < N; ++i)
      g[i] += phi[i];

    // A1.4.1
    bool binding[N];
    for (size_t i = 0; i < N; ++i)
      binding[i] = ( xInit[i] <= l[i] && g[i] > 0.0f ) || ( xInit[i] >= u[i] && g[i] < 0.0f );

    // A1.4.2
    float hRed[N][N];
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
	if (i == j)
	  hRed[i][j] = Gamma[i][j];
	else
	  hRed[i][j] = binding[i] || binding[j]? 0.0f : Gamma[i][j];

    float hRedInv[N][N];
    nstx_matrixInvert(N, hRed, hRedInv);

    float d[N];
    memset(d, 0, sizeof(d));
    nstx_matrixMult2d1d(N, N, hRedInv, g, d);
    for (size_t i = 0; i < N; ++i)
      d[i] *= -1.0f;

    float fLow = FLT_MAX;
    float xLow[N];
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
    for (size_t i = 0; i < N; ++i)
      xInit[i] = xLow[i];

    if (fLow < fBest) {
      fBest = fLow;
      for (size_t i = 0; i < N; ++i)
	xBest[i] = xInit[i];
    }
  }

  for (size_t i = 0; i < N; ++i) 
    xOut[i] += lambda * xBest[i];
}
