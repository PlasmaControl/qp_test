#include <stddef.h> //for size_t

void nstx_matrixMult1d2d(size_t const N, size_t const M, float const A[VEPN], float const B[VEPN][VEPN], float out[VEPN]);
void nstx_matrixMult2d1d(size_t const N, size_t const M, float const A[VEPN][VEPN], float const B[VEPN], float out[VEPN]);
void nstx_matrixMult2d2d(size_t const N, size_t const M, size_t const P, float const A[VEPN][VEPN], float const B[VEPN][VEPN], float out[VEPN][VEPN]);
void nstx_matrixInvert(size_t const N, float const in[VEPN][VEPN], float out[VEPN][VEPN]);
