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
	      float const eps, size_t const nIter,
	      float xOut[restrict N],
	      float yOut[restrict M]) {

	float fastMax(float const a, float const b) {
		// GCC isn't optimizing fmaxf correctly, so inline here
		return a > b? a : b;
	}

	float infNorm(size_t const N, float const v[N]) {
		float temp = 0.0f;
		for (size_t i = 0; i < N; ++i)
			temp = fastMax(temp, fabsf(v[i]));
		return temp;
	}

	float const alpha1 = 1.0f - alpha;

	float const qNorm = infNorm(N, q);

	float AT[N][M];
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < M; ++j)
			AT[i][j] = A[j][i];

	float x[N];
	for (size_t i = 0; i < N; ++i)
		x[i] = xOut[i];

	float y[M];
	for (size_t i = 0; i < M; ++i)
		y[i] = yOut[i];

	float z[M];
	memset(z, 0, sizeof(z));
	nstx_matrixMult2d1d(M, N, A, x, z);

	for (size_t i = 0; i < nIter; ++i) {
		float rhoZ[M];
		for (size_t j = 0; j < M; ++j)
			rhoZ[j] = rho * z[j] - y[j];

		float ATRhoZ[N];
		memset(ATRhoZ, 0, sizeof(ATRhoZ));
		nstx_matrixMult2d1d(N, M, AT, rhoZ, ATRhoZ);

		float w[N];
		for (size_t j = 0; j < N; ++j)
			w[j] = sigma * x[j] - q[j] + ATRhoZ[j];

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
		float zNextNorm = 0.0f;
		for (size_t j = 0; j < M; ++j) {
			float const zTemp = clamp(l[j], u[j], alpha * zHat[j] + alpha1 * z[j] + y[j] / rho);
			zNextNorm = fastMax(zNextNorm, fabsf(zTemp));
			zNext[j] = zTemp;
		}

		float yNext[M];
		for (size_t j = 0; j < M; ++j)
			yNext[j] = y[j] + rho * (alpha * zHat[j] + alpha1 * z[j] - zNext[j]);

		for (size_t j = 0; j < N; ++j) x[j] = xNext[j];
		for (size_t j = 0; j < M; ++j) y[j] = yNext[j];
		for (size_t j = 0; j < M; ++j) z[j] = zNext[j];

		continue;
		float AXNext[M];
		memset(AXNext, 0, sizeof(AXNext));
		nstx_matrixMult2d1d(M, N, A, xNext, AXNext);
		float const AXNextNorm = infNorm(M, AXNext);

		float PXNext[N];
		memset(PXNext, 0, sizeof(PXNext));
		nstx_matrixMult2d1d(N, N, P, xNext, PXNext);
		float const PXNextNorm = infNorm(N, PXNext);

		float ATYNext[N];
		memset(ATYNext, 0, sizeof(ATYNext));
		nstx_matrixMult2d1d(N, M, AT, yNext, ATYNext);
		float const ATYNextNorm = infNorm(N, ATYNext);

		float const epsP = eps + eps * fastMax(AXNextNorm, zNextNorm);
		float const epsD = eps + eps * fastMax(fastMax(PXNextNorm, ATYNextNorm), qNorm);

		float rpVec[M];
		for (size_t j = 0; j < M; ++j)
			rpVec[j] = AXNext[j] - zNext[j];
		float const rpNorm = infNorm(M, rpVec);
	}

	for (size_t i = 0; i < N; ++i)
		xOut[i] = x[i];
	for (size_t i = 0; i < M; ++i)
		yOut[i] = y[i];
}

void mpc_setup(size_t const nZ, size_t const nU, size_t const nLook,
	       float const A[nZ][nZ],
	       float const B[nZ][nU],
	       float const Q[nZ][nZ],
	       float const R[nU][nU],
	       float const sigma, float const rho, float const delta_t,
	       float E[nZ][nZ*nLook],
	       float F[nZ*nLook][nU*nLook],
	       float P[nU*nLook][nU*nLook],
	       float G[nU*nLook][nU*nLook],
	       float Ac[nU*(2*nLook-1)][nU*nLook],
	       float QHat[nZ*nLook],
	       float RHat[nU*nLook]
	      ) {
	size_t const nZL = nZ * nLook;
	size_t const nUL = nU * nLook;

	float Atemp[nLook][nZ][nZ];
	for (size_t i = 0; i < nZ; ++i)
		for (size_t j = 0; j < nZ; ++j)
			Atemp[0][i][j] = A[i][j];
	for (size_t i = 1; i < nLook; ++i)
		nstx_matrixMult2d2d(nZ, nZ, nZ, A, Atemp[i-1], Atemp[i]);

	for (size_t i = 0; i < nLook; ++i)
		for (size_t j = 0; j < nZ; ++j)
			for (size_t k = 0; k < nZ; ++k)
				E[j][k + i*nZ] = Atemp[i][j][k];

	for (size_t i = 0; i < nLook; ++i)
		for (size_t j = 0; j <= i; ++j) {
			float Ftemp[nZ][nU];
			memset(Ftemp, 0, sizeof(Ftemp));
			if (i == j)
				for (size_t x = 0; x < nZ; ++x)
					for (size_t y = 0; y < nU; ++y)
						Ftemp[x][y] = B[x][y];
			else
				nstx_matrixMult2d2d(nZ, nZ, nU, Atemp[i - 1], B, Ftemp);
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

	for (size_t i = 0; i < nUL; ++i)
		for (size_t j = 0; j < nUL; ++j)
			Ac[i][j] = (i == j)? 1.0f : 0.0f;

	qp_setup(nUL, nUL, P, Ac, sigma, rho, G);
}

void mpc_solve(size_t const nZ, size_t const nU, size_t nLook,
	       size_t const rho, size_t const sigma, size_t const alpha, 
	       size_t const epsilon, size_t const nIter,
	       float const z[restrict nZ],
	       float const r[restrict nZ],
	       float const uMin[restrict nU],
	       float const uMax[restrict nU],
	       float const deltauMin[restrict nU],
	       float const deltauMax[restrict nU],
	       float const uref[restrict nU],
	       float const E[restrict nZ][nZ * nLook],
	       float const F[restrict nZ * nLook][nU * nLook],
	       float const P[restrict nU * nLook][nU * nLook],
	       float const G[restrict nU * nLook][nU * nLook],
	       float const Ac[restrict nU * (2*nLook-1)][nU * nLook],
	       float const QHat[restrict nZ * nLook],
	       float const RHat[restrict nU * nLook],
	       float uHat[restrict nU * nLook],
	       float lambda[restrict nU * nLook]
	      ) {
	size_t const nZL = nZ * nLook;
	size_t const nUL = nU * nLook;

	float rHat[nZL];
	for (size_t i = 0; i < nLook; ++i)
		for (size_t j = 0; j < nZ; ++j)
			rHat[i * nZ + j] = r[j];

	float uHatMin[nUL];
	float uHatMax[nUL];
	for (size_t i = 0; i < nLook; ++i)
		for (size_t j = 0; j < nU; ++j) {
			uHatMin[i * nU + j] = uMin[j];
			uHatMax[i * nU + j] = uMin[j];
		}

	float fTemp[nZL];
	memset(fTemp, 0, sizeof(fTemp));
	nstx_matrixMult1d2d(nZ, nZL, z, E, fTemp);
	for (size_t i = 0; i < nZL; ++i) {
		fTemp[i] -= rHat[i];
		fTemp[i] *= QHat[i];
	}
	float f[nUL];
	memset(f, 0, sizeof(f));
	nstx_matrixMult1d2d(nZL, nUL, fTemp, F, f);

	qp_solve(nUL, nU, G, P, Ac, rho, sigma, alpha, f, uMin, uMax, epsilon, nIter, uHat, lambda);
}
