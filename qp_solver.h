#include <stddef.h> //for size_t

/**************************************************
    Solve QP of the form

    min_x  1/2 x^T P x + q^T x
    subject to l <= Ax <= u


    Parameters
    ----------
    G : ndarray (n,n)
        reduced KKT matrix, output of qp_setup
    P : ndarray (n,n)
        hessian matrix
    A : ndarray (m,n)
        constraint matrix
    rho : float
        step size
    sigma : float
        normalization parameter
    alpha : float
        relaxation parameter
    q : ndarray (n,)
        gradient vector
    l, u : ndarray (m,)
        upper and lower bounds on Ax
    x0 : ndarray (n,)
        initial guess for solution
    y0 : ndarray (m,)
        initial guess for lagrange multipliers
    eps : float
        stopping tolerance
    maxiter : int
        maximum number of iterations
    verbose : bool
        whether to display progress

    Returns
    -------
    x : ndarray (n,)
        solution vector
    y : ndarray (m,)
        lagrange multipliers
    r_prim, r_dual : float
        primal and dual residuals
**************************************************/
void qp_solve(size_t const N, size_t const M,
	      float const G[N][N], float const P[N][N], float const A[M][N], 
	      float const rho, float const sigma, float const alpha, 
	      float const q[N], 
	      float const l[M], float const u[M], 
	      float const x0[N], 
	      float const eps, size_t const maxiter,
	      float xOut[N], float yOut[M], float r[2]);
