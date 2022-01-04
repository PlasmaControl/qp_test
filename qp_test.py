import numpy as np


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


"""problem dimensions"""
n = 5 # number of variables
m = 4 # number of constraints

"""Uncomment below to try new inputs"""

# P = np.random.random((n,n))
# P = P.T @ P
# A = np.eye(m,n)
# q = np.random.random(n)
# l = -np.random.random(m)
# u = np.random.random(m)

# print(P)
# print(A)
# print(q)
# print(l)
# print(u)


"""or use these hard-coded values"""

P = np.array([[1.63729889, 1.49800623, 0.66972223, 1.16520369, 1.00871733],
 [1.49800623, 2.30753699, 1.00125695, 1.29220823, 1.59194631],
 [0.66972223, 1.00125695, 1.16973938, 0.60180993, 0.81632789],
 [1.16520369, 1.29220823, 0.60180993, 1.33521535, 1.24619392],
 [1.00871733, 1.59194631, 0.81632789, 1.24619392, 1.4141925 ]])
A = np.array([[1., 0., 0., 0., 0.],
 [0., 1., 0., 0., 0.],
 [0., 0., 1., 0., 0.],
 [0., 0., 0., 1., 0.]])
q = np.array([0.87613838, 0.7393823,  0.55419765, 0.59420561, 0.26704384])
l = np.array([-0.37471228, -0.80018154, -0.47095637, -0.11762543])
u = np.array([0.21614515, 0.50421568, 0.35092037, 0.95366794])

x0 = np.zeros(n)
y0 = np.zeros(m)

"""Solver parameters"""
sigma = 1e-6
rho = 6
alpha = 1.6
maxiter=1000
eps=1e-6


G = qp_setup(P,A,rho,sigma)
xf, yf, k, r_prim, r_dual = qp_solve(G,P,A,rho,sigma,alpha,q,l,u,x0,y0, eps, maxiter)

print("Python implementation: ")
print("x=", xf)
print("y=", yf)
print()

import ctypes
import os
libname=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "libqp_solver.so")
c_lib=ctypes.CDLL(libname)

c_lib.qp_solve.argtypes = [
    ctypes.c_size_t, #N
    ctypes.c_size_t, #M
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #G
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #P
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #A
    ctypes.c_float, #rho    
    ctypes.c_float, #sigma
    ctypes.c_float, #alpha
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #q
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #l
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #u
    ctypes.c_float, #eps
    ctypes.c_size_t, #maxiter
    # we want to change the value of an input array, so have to 
    # convert to a pointer (C passes by value, so we need to supply
    # an address to copy rather than an array of data)
    ctypes.POINTER(ctypes.c_double), #xOut
    ctypes.POINTER(ctypes.c_double)] #yOut
c_lib.qp_solve.restype = ctypes.c_void_p #void return (answer goes into xOut)

xout=x0.copy().astype(np.float32)
yout=y0.copy().astype(np.float32)
c_lib.qp_solve(len(q),
               len(A),
               G.astype(np.float32),
               P.astype(np.float32),
               A.astype(np.float32),
               rho, sigma, alpha,
               q.astype(np.float32),
               l.astype(np.float32),
               u.astype(np.float32),
               eps, maxiter,
               xout.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               yout.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

print("C implementation of qp_solver: ")
print("x=", xout)
print("y=", yout)

c_lib.qp_setup.argtypes = [
    ctypes.c_size_t, #N
    ctypes.c_size_t, #M
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #P
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #A
    ctypes.c_float, #sigma
    ctypes.c_float, #rho
    # we want to change the value of an input array, so have to 
    # convert to a pointer (C passes by value, so we need to supply
    # an address to copy rather than an array of data)
    ctypes.POINTER(ctypes.c_double)] #G

G=np.zeros_like(P).astype(np.float32)
c_lib.qp_setup(len(P),len(A),
               P.astype(np.float32),A.astype(np.float32),
               sigma,rho,
               G.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
xf, yf, k, r_prim, r_dual = qp_solve(G,P,A,rho,sigma,alpha,q,l,u,x0,y0, eps, maxiter)

print("C implementation of qp_setup: ")
print("x=", xf)
print("y=", yf)
