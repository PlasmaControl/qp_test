#include "qp_solver.h"
#include "nstx_math.h"
#include <math.h>
#include <stddef.h> //for size_t
#include <string.h>
#include <stdio.h>

void qp_setup(size_t const N, size_t const M, float const P[N][N], float const A[M][N], float const sigma, float const rho, float G[N][N]) {
	float AT[N][M];
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			AT[i][j] = A[j][i] * rho;

	float rhoATA[N][N];
	memset(rhoATA, 0, sizeof(rhoATA));
	nstx_matrixMult2d2d(N, M, N, AT, A, rhoATA);

	float H[N][N];
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			H[i][j] = P[i][j] + rhoATA[i][j];
	for (size_t i = 0; i < N; ++i)
		H[i][i] += sigma;

	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			G[i][j] = 0.0f;
	nstx_matrixInvert(N, H, G);
}

void qp_solve(size_t const N, size_t const M,
	      float const G[restrict N][N], float const P[restrict N][N], float const A[restrict M][N],
	      float const rho, float const sigma, float const alpha,
	      float const q[restrict N],
	      float const l[restrict M], float const u[restrict M],
	      size_t const nIter,
	      float xOut[restrict N],
	      float yOut[restrict M],
	      float residual[restrict 2]) {

	float fastMax(float const a, float const b) {
		// GCC isn't optimizing fmaxf correctly, so inline here
		return a > b? a : b;
	}

	float const alpha1 = 1.0f - alpha;

	float * x = xOut;
	float * y = yOut;

	float z[M];
	memset(z, 0, sizeof(z));
	nstx_matrixMult2d1d(M, N, A, x, z);

	for (size_t i = 0; i < nIter; ++i) {
		float rhoZ[M];
		for (size_t j = 0; j < M; ++j)
			rhoZ[j] = rho * z[j] - y[j];

		float ARhoZ[N];
		memset(ARhoZ, 0, sizeof(ARhoZ));
		nstx_matrixMult1d2d(M, N, rhoZ, A, ARhoZ);

		float w[N];
		for (size_t j = 0; j < N; ++j)
			w[j] = sigma * x[j] - q[j] + ARhoZ[j];

		float xHat[N];
		memset(xHat, 0, sizeof(xHat));
		nstx_matrixMult2d1d(N, N, G, w, xHat);

		float zHat[M];
		memset(zHat, 0, sizeof(zHat));
		nstx_matrixMult2d1d(M, N, A, xHat, zHat);

		float xNext[N];
		for (size_t j = 0; j < N; ++j)
			xNext[j] = alpha * xHat[j] + alpha1 * x[j];

		float zNext[M];
		for (size_t j = 0; j < M; ++j)
			zNext[j] = clamp(l[j], u[j], alpha * zHat[j] + alpha1 * z[j] + y[j] / rho);

		float yNext[M];
		for (size_t j = 0; j < M; ++j)
			yNext[j] = y[j] + rho * (alpha * zHat[j] + alpha1 * z[j] - zNext[j]);

		for (size_t j = 0; j < N; ++j) x[j] = xNext[j];
		for (size_t j = 0; j < M; ++j) y[j] = yNext[j];
		for (size_t j = 0; j < M; ++j) z[j] = zNext[j];
	}

	float infNorm(size_t const N, float const v[N]) {
		float temp = 0.0f;
		for (size_t i = 0; i < N; ++i)
			temp = fastMax(temp, fabsf(v[i]));
		return temp;
	}

	float resP[M];
	memset(resP, 0, sizeof(resP));
	nstx_matrixMult2d1d(M, N, A, x, resP);
	for (size_t i = 0; i < M; ++i)
		resP[i] -= z[i];
	residual[0] = infNorm(M, resP);

	float Px[N];
	float yA[N];
	float resD[N];
	memset(Px, 0, sizeof(Px));
	memset(yA, 0, sizeof(yA));
	nstx_matrixMult2d1d(N, N, P, x, Px);
	nstx_matrixMult1d2d(N, M, y, A, yA);
	for (size_t i = 0; i < N; ++i)
		resD[i] = Px[i] + q[i] + yA[i];
	residual[1] = infNorm(N, resD);
}

void mpc_setup(size_t const nZ, size_t const nU, size_t const nLook,
	       float const A[nZ][nZ],
	       float const B[nZ][nU],
	       float const Q[nZ][nZ],
	       float const R[nU][nU],
	       float const sigma, float const rho, float const deltaT,
	       float E[nZ*nLook][nZ],
	       float F[nZ*nLook][nU*nLook],
	       float P[nU*nLook][nU*nLook],
	       float G[nU*nLook][nU*nLook],
	       float Ac[nU*(2*nLook-1)][nU*nLook],
	       float QHat[nZ*nLook],
	       float RHat[nU*nLook]
	      ) {
	size_t const nZL = nZ * nLook;
	size_t const nUL = nU * nLook;
	size_t const nUL2 = nU * (2 * nLook - 1);

	for (size_t i = 0; i < nZ; ++i)
		for (size_t j = 0; j < nZ; ++j)
			E[i][j] = A[i][j];
	for (size_t i = nZ; i < nZL; i += nZ)
		nstx_matrixMult2d2d(nZ, nZ, nZ, A, &E[i - nZ], &E[i]);

	for (size_t i = 0; i < nLook; ++i)
		for (size_t x = 0; x < nZ; ++x)
			for (size_t y = 0; y < nU; ++y)
				F[i * nZ + x][i * nU + y] = B[x][y];
	for (size_t i = 1; i < nLook; ++i)
		for (size_t j = 0; j < i; ++j) {
			float Ftemp[nZ][nU];
			memset(Ftemp, 0, sizeof(Ftemp));
			nstx_matrixMult2d2d(nZ, nZ, nU, &E[nZ * (i - j - 1)], B, Ftemp);
			for (size_t x = 0; x < nZ; ++x)
				for (size_t y = 0; y < nU; ++y)
					F[i * nZ + x][j * nU + y] = Ftemp[x][y];
		}

	for (size_t i = 0; i < nLook; ++i)
		for (size_t j = 0; j < nZ; ++j)
			QHat[i * nZ + j] = Q[j][j];

	for (size_t i = 0; i < nLook; ++i)
		for (size_t j = 0; j < nU; ++j)
			RHat[i * nU + j] = R[j][j];

	float Ptemp[nUL][nZL];
	for (size_t i = 0; i < nUL; ++i)
		for (size_t j = 0; j < nZL; ++j)
			Ptemp[i][j] = F[j][i] * QHat[j];
	nstx_matrixMult2d2d(nUL, nZL, nUL, Ptemp, F, P);
	for (size_t i = 0; i < nUL; ++i)
		P[i][i] += RHat[i];

	// Zero entire matrix to start
	for (size_t i = 0; i < nUL2; ++i)
		for (size_t j = 0; j < nUL; ++j)
			Ac[i][j] = 0.0f;

	// First nUL rows of Ac is normal identity
	for (size_t i = 0; i < nUL; ++i)
		Ac[i][i] = 1.0f;

	// Remaining rows have two diagonals
	for (size_t i = nUL; i < nUL2; ++i) {
		Ac[i][i - nUL] = -1.0f / deltaT;
		Ac[i][i - 2] = 1.0f / deltaT;
	}

	qp_setup(nUL, nUL2, P, Ac, sigma, rho, G);
}

void mpc_solve(size_t const nZ, size_t const nU, size_t nLook,
	       size_t const rho, size_t const sigma, size_t const alpha, 
	       size_t const nIter,
	       float const z[restrict nZ],
	       float const rHat[restrict nZ * nLook],
	       float const uHatMin[restrict nU * nLook],
	       float const uHatMax[restrict nU * nLook],
	       float const uHatMinDelta[restrict nU * (nLook - 1)],
	       float const uHatMaxDelta[restrict nU * (nLook - 1)],
	       float const uHatRef[restrict nU * nLook],
	       float const E[restrict nZ * nLook][nZ],
	       float const F[restrict nZ * nLook][nU * nLook],
	       float const P[restrict nU * nLook][nU * nLook],
	       float const G[restrict nU * nLook][nU * nLook],
	       float const Ac[restrict nU * (2 * nLook - 1)][nU * nLook],
	       float const QHat[restrict nZ * nLook],
	       float const RHat[restrict nU * nLook],
	       float uHat[restrict nU * nLook],
	       float lambda[restrict nU * (2 * nLook - 1)]
	      ) {
	size_t const nZL = nZ * nLook;
	size_t const nUL = nU * nLook;
	size_t const nUL2 = nU * (2 * nLook - 1);

	float lower[nUL2];
	float upper[nUL2];
	for (size_t i = 0; i < nUL; ++i) {
		lower[i] = uHatMin[i];
		upper[i] = uHatMax[i];
	}
	for (size_t i = nUL; i < nUL2; ++i) {
		lower[i] = uHatMinDelta[i - nUL];
		upper[i] = uHatMaxDelta[i - nUL];
	}

	float fTemp[nZL];
	memset(fTemp, 0, sizeof(fTemp));
	nstx_matrixMult2d1d(nZL, nZ, E, z, fTemp);
	for (size_t i = 0; i < nZL; ++i) {
		fTemp[i] -= rHat[i];
		fTemp[i] *= QHat[i];
	}

	float f[nUL];
	memset(f, 0, sizeof(f));
	nstx_matrixMult1d2d(nZL, nUL, fTemp, F, f);
	for (size_t i = 0; i < nUL; ++i) {
		f[i] -= RHat[i] * uHatRef[i];
		f[i] *= 2.0f;
	}

	float residual[2] = {0};
	qp_solve(nUL, nU, G, P, Ac, rho, sigma, alpha, f, lower, upper, nIter, uHat, lambda, residual);
}
