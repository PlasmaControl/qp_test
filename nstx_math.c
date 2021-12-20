#include "nstx_math.h"
#include <stddef.h>
#include <math.h>

float clamp(float const min, float const max, float const val) {
	float const t = val < min? min : val;
	return t > max? max : t;
}

void nstx_matrixMult1d2d(size_t const N, size_t const M, float const A[N], float const B[N][M], float out[M]) {
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			out[i] += A[j] * B[j][i];
}

void nstx_matrixMult2d1d(size_t const N, size_t const M, float const A[N][M], float const B[M], float out[M]) {
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			out[i] += A[i][j] * B[j];
}

void nstx_matrixMult2d2d(size_t const N, size_t const M, size_t const P, float const A[N][M], float const B[M][P], float out[N][P]) {
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			for (size_t k = 0; k < P; ++k)
				out[i][k] += A[i][j] * B[j][k];
}

void nstx_matrixInvert(size_t const N, float const in[N][N], float out[N][N]) {
	// augment matrix with identity
	size_t const COLS = 2 * N;
        float b[N][COLS];
        for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j) {
                        b[i][j] = in[i][j];
                        b[i][j+N] = (i == j);
                }

	// Gaussian Elimination with Partial Pivoting
        for (size_t i = 0; i < N; ++i) {
		// pivoting (row swapping)
		for (size_t j = i+1; j < N; ++j)
			if (fabsf(b[j][i]) > fabsf(b[i][i]))
				for (size_t k = 0; k < COLS; ++k) {
					float const temp = b[i][k];
					b[i][k] = b[j][k];
					b[j][k] = temp;
				}

                float const div = 1.0f / b[i][i];
                //for (size_t j = i; j <= N + i; ++j)
		for (size_t j = 0; j < COLS; ++j)
                        b[i][j] *= div;

                for (size_t m = 0; m < N; ++m) {
                        if (m == i)
                                continue;

                        float const x = b[m][i];
                        //for (size_t j = i; j <= N + i; ++j)
                        for (size_t j = 0; j < COLS; ++j)
                                b[m][j] -= x * b[i][j];
                }
        }

	// extract inverse matrix
        for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                        out[i][j] = b[i][j+N];
}


