#include "qp_solver.h"
#include "nstx_math.h"
#include <math.h>
#include <stddef.h> //for size_t
#include <string.h>
#include <stdio.h>

void pVec(char const * name, size_t const N, float x[N]) {
	printf("Vec %s:\n", name);
	for (size_t i = 0; i < N; ++i)
		printf("%f\t", x[i]);
	printf("\n");
}

void pMat(char const * name, size_t const X, size_t const Y, float x[X][Y]) {
	printf("Mat %s:\n", name);
	for (size_t i = 0; i < X; ++i) {
		for (size_t j = 0; j < Y; ++j)
			printf("%f\t", x[i][j]);
		printf("\n");
	}
}

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
	       size_t const nIter,
	       float const rho, float const sigma, float const alpha,
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
	qp_solve(nUL, nUL2, G, P, Ac, rho, sigma, alpha, f, lower, upper, nIter, uHat, lambda, residual);
}


#define N 5
#define M 4
#define NLOOK 3
int main () {
	float P[N][N] = {{1.63729889f, 1.49800623f, 0.66972223f, 1.16520369f, 1.00871733f},
			 {1.49800623f, 2.30753699f, 1.00125695f, 1.29220823f, 1.59194631f},
			 {0.66972223f, 1.00125695f, 1.16973938f, 0.60180993f, 0.81632789f},
			 {1.16520369f, 1.29220823f, 0.60180993f, 1.33521535f, 1.24619392f},
			 {1.00871733f, 1.59194631f, 0.81632789f, 1.24619392f, 1.4141925f }};
	float A[M][N] = {{1.f, 0.f, 0.f, 0.f, 0.f},
			 {0.f, 1.f, 0.f, 0.f, 0.f},
			 {0.f, 0.f, 1.f, 0.f, 0.f},
			 {0.f, 0.f, 0.f, 1.f, 0.f}};

	float q[N] = {0.87613838f, 0.7393823f,  0.55419765f, 0.59420561f, 0.26704384f};
	float l[M] = {-0.37471228f, -0.80018154f, -0.47095637f, -0.11762543f};
	float u[M] = {0.21614515f, 0.50421568f, 0.35092037f, 0.95366794f};

	float x0[N] = {0};
	float y0[M] = {0};
	float residual[2] = {0};

	float sigma = 1.e-6;
	float rho = 6.f;
	float alpha = 1.6f;
	size_t maxiter=3;

	float G[N][N] = {0};

	qp_setup(N, M, P, A, sigma, rho, G);
	qp_solve(N, M, G, P, A, rho, sigma, alpha, q, l, u, maxiter, x0, y0, residual);

	printf("Compare to running python qp_test.py:\n");
	pMat("G",N,N,G);
	pVec("x0",N,x0);
	pVec("y0",M,y0);
	pVec("residual",2,residual);
}
	/*
	float A[NZ][NZ] = {
		{ 1.f, 0.02f},
		{-0.02f, 1.f}
	};
	float B[NZ][NU] = {
		{0.f  },
		{0.02f}
	};
	float Q[NZ][NZ] = {
		{1.0f},
		{0.0f,1.0f}
	};
	float R[NU][NU] = {
		{1.0f}
	};

	float E[NZ*NLOOK][NZ] = {0};
	float F[NZ*NLOOK][NU*NLOOK] = {0};
	float P[NU*NLOOK][NU*NLOOK] = {0};
	float G[NU*NLOOK][NU*NLOOK] = {0};
	float Ac[NU*(2*NLOOK-1)][NU*NLOOK] = {0};
	float QHat[NZ*NLOOK] = {0};
	float RHat[NU*NLOOK] = {0};

	mpc_setup(nZ, nU, nLook, A, B, Q, R, sigma, rho, deltaT, E, F, P, G, Ac, QHat, RHat);

	pMat("A", NZ, NZ, A);
	pMat("B", NZ, NU, B);
	pMat("Q", NZ, NZ, Q);
	pMat("R", NU, NU, R);
	pMat("E", nZ*nLook, NZ, E);
	pMat("F", nZ*nLook, nU*nLook, F);
	pMat("P", nU*nLook, nU*nLook, P);
	pMat("G", nU*nLook, nU*nLook, G);
	pMat("Ac", nU*(2*nLook-1), nU*nLook, Ac);
	pVec("QHat", nZ*nLook, QHat);
	pVec("RHat", nU*nLook, RHat);

	float alpha = 1.6f;
	size_t nIter = 3;
	float z[NZ] = { 0.0f, 0.01f };
	float rHat[NZ * NLOOK] = {0};
	float uHatMin[NU * NLOOK] = {-1.0f, -1.0f, -1.0f};
	float uHatMax[NU * NLOOK] = {1.0f, 1.0f, 1.0f};
	float uHatMinDelta[NU * (NLOOK - 1)] = {-1.0f, -1.0f};
	float uHatMaxDelta[NU * (NLOOK - 1)] = {1.0f, 1.0f};
	float uHatRef[NU * NLOOK] = {0};
	float uHat[NU * NLOOK] = {0};
	float lambda[NU * (2 * NLOOK - 1)] = {0};

	mpc_solve(nZ, nU, nLook, nIter, rho, sigma, alpha, z, rHat, uHatMin, uHatMax, uHatMinDelta, uHatMaxDelta, uHatRef, E, F, P, G, Ac, QHat, RHat, uHat, lambda);

	pVec("z", NZ, z);
	pVec("rHat", NU * NLOOK, rHat);
	pVec("uHatMin", NU * NLOOK, uHatMin);
	pVec("uHatMax", NU * NLOOK, uHatMax);
	pVec("uHatMinDelta", NU * (NLOOK - 1), uHatMinDelta);
	pVec("uHatMaxDelta", NU * (NLOOK - 1), uHatMaxDelta);
	pVec("uHatRef", NU * NLOOK, uHatRef);
	pMat("E", nZ*nLook, NZ, E);
	pMat("F", nZ*nLook, nU*nLook, F);
	pMat("P", nU*nLook, nU*nLook, P);
	pMat("G", nU*nLook, nU*nLook, G);
	pMat("Ac", nU*(2*nLook-1), nU*nLook, Ac);
	pVec("QHat", nZ*nLook, QHat);
	pVec("RHat", nU*nLook, RHat);
	pVec("uHat", NU * NLOOK, uHat);
	pVec("lambda", NU * (2 * NLOOK - 1), lambda);

	return 0;
}
	*/
