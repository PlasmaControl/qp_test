import numpy as np
from scipy import linalg

def qp_setup(P,A,rho,sigma):
    """Form QP reduced KKT matrix

    Parameters
    ----------
    P : ndarray (n,n)
        hessian matrix
    A : ndarray (m,n)
        constraint matrix
    rho : float
        step size
    sigma : float
        normalization parameter

    Returns
    -------
    G : ndarray (n,n)
        inverse of KKT matrix
    """

    G = P + sigma*np.eye(P.shape[0]) + rho*A.T @ A
    return np.linalg.inv(G)


def qp_solve(G,P,A,rho,sigma,alpha,q,l,u,x0,y0, eps, maxiter, verbose=False):
    """Solve QP of the form

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
    """
    
    xk = x0
    zk = A@x0
    yk = y0
    
    r_prim = np.inf
    r_dual = np.inf
    eps_prim = eps
    eps_dual = eps
    k = 1
    while (k<maxiter):
        k += 1
        w = sigma*xk - q + A.T @ (rho*zk-yk)
        xhk1 = G @ w
        zhk1 = A @ xhk1
        xk1 = alpha*xhk1 + (1-alpha)*xk
        zk1 = np.clip(alpha*zhk1 + (1-alpha)*zk + yk/rho, l, u)
        yk1 = yk + rho*(alpha*zhk1 + (1-alpha)*zk - zk1)

        eps_prim = eps + eps*np.max([np.max(abs(A@xk1)), np.max(abs(zk1))])
        eps_dual = eps + eps*np.max([np.max(abs(P@xk1)), np.max(abs(A.T@yk1)), np.max(abs(q))])
        
        r_prim = np.max(np.abs(A@xk1 - zk1))
        r_dual = np.max(np.abs(P @ xk1 + q + A.T @ yk1))
        xk = xk1
        zk = zk1
        yk = yk1
        if (k%25 == 0) or (k ==1):
            if verbose:
                print("{:4d}   {:.3e}   {:.3e}".format(k, r_prim, r_dual))
            if ((r_prim < eps_prim) and (r_dual < eps_dual)):
                break
    return xk, yk, k, r_prim, r_dual

def mpc_setup(Nlook, A, B, Q, R, sigma, rho, delta_t):
    """setup mpc problem

    Parameters
    ----------
    Q, R: ndarray
        state and control weight matrices
    Nlook : int
        number of steps in the mpc lookahead

    """
    # E = [A, A**2, A**3,...]
    # F = [B 0 0 ...]
    #     [AB B 0 0 ...]
    #     [A^2B AB B 0 0 ...]
    E = np.vstack([np.linalg.matrix_power(A, i + 1) for i in range(Nlook)])
    F = []
    for i in range(Nlook):
        Frow = np.hstack(
            [np.linalg.matrix_power(A, j) @ B for j in range(i + 1)][::-1]
        )
        F.append(
            np.hstack([Frow, np.zeros((A.shape[0], (Nlook - i - 1) * B.shape[1]))])
        )

    F = np.vstack(F)
    nu = B.shape[1]
    Aineq_bound_u = np.eye(Nlook * nu)
    Aineq_rate_u = np.zeros((nu*(Nlook-1),nu*Nlook))
    for i in range(nu*(Nlook-1)):
        Aineq_rate_u[i][i]=-1/delta_t
        Aineq_rate_u[i][i+nu]=1/delta_t
    Aineq = np.vstack([Aineq_bound_u, Aineq_rate_u])

    # expand cost matrices
    Qhat = np.array([Q] * Nlook)

    # we need the lqr addition for the mpc algo to work for control properly
    # but keeping like this for now
    if False:
        import scipy
        lqr_P=scipy.linalg.solve_discrete_are(A,B,Q,R)    
        Qhat[-1]=lqr_P #LQR to cut off the time

    Rhat = np.array([R] * Nlook)
    
    Qhat=linalg.block_diag(*Qhat)
    Rhat=linalg.block_diag(*Rhat)
    H = F.T @ Qhat @ F + Rhat
    # symmetrize for accuracy
    H = (H + H.T) / 2
    lu = np.ones(Aineq.shape[0])
    # note in the spec H --> P    
    G = qp_setup(H,Aineq,rho,sigma)

    return (E, F, H, G, Aineq, Qhat, Rhat)

def mpc_solve(Nlook,
              r, z, uMin, uMax,
              uref, delta_uMin, delta_uMax,
              E, F, P, G, Ac, Qhat, Rhat,
              rho, sigma, alpha, epsilon, nIter,
              x0, y0):
    umin_hat = np.tile(uMin, (Nlook,))
    umax_hat = np.tile(uMax, (Nlook,))
    delta_umin_hat = np.tile(delta_uMin, (Nlook-1,))
    delta_umax_hat = np.tile(delta_uMax, (Nlook-1,))
    lower=np.concatenate([umin_hat,delta_umin_hat])
    upper=np.concatenate([umax_hat,delta_umax_hat])
    uref_hat = np.tile(uref, (Nlook,))
    zref_hat = np.tile(r, (Nlook,))

    # note that factor of 2 to be consistent with how osqp/we define it
    f = 2* ( (z.T @ E.T - zref_hat) @ Qhat @ F - Rhat @ uref_hat )
    xf, yf, k, r_prim, r_dual = qp_solve(G,P,Ac,rho,sigma,alpha,f,lower,upper,x0,y0, epsilon, nIter)
    return (xf, yf)

