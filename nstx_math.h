#include <stddef.h> //for size_t

void nstx_matrixMult1d2d(size_t const N, size_t const M, float const A[N], float const B[N][M], float out[M]);
void nstx_matrixMult2d1d(size_t const N, size_t const M, float const A[N][M], float const B[M], float out[M]);
void nstx_matrixMult2d2d(size_t const N, size_t const M, size_t const P, float const A[N][M], float const B[M][P], float out[N][P]);
void nstx_matrixInvert(size_t const N, float const in[N][N], float out[N][N]);
