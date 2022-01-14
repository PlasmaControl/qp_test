import numpy as np
import scipy
import ctypes
import os
import mpc_qp_helpers
import python_c_helpers

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
maxiter=3

G = mpc_qp_helpers.qp_setup(P,A,rho,sigma)
print("Python implementation of qp_setup:")
print("G:")
print(G)

G_C=python_c_helpers.qp_setup(P,A,rho,sigma)

print("C implementation of qp_setup:")
print("G:")
print(G_C)

print("\n\n\n\n")
try:
    import osqp
    qp=osqp.OSQP()
    qp.setup(P=scipy.sparse.csc_matrix(P),
             q=q,
             A=scipy.sparse.csc_matrix(A),
             l=l,
             u=u,
             verbose=False)
    print("OSQP implementation of qp_solve: ")
    results = qp.solve()
    print("x=", results.x)
    print("y=", results.y)
    print("residual=",[results.info.pri_res,results.info.dua_res])
    print()
except:
    pass

xf, yf, k, r_prim, r_dual = mpc_qp_helpers.qp_solve(G=G, P=P, q=q, A=A,
                                                    l=l, u=u,
                                                    rho=rho, sigma=sigma, alpha=alpha,
                                                    x0=x0, y0=y0, maxiter=maxiter)
print("Python implementation of qp_solve: ")
print("x=", xf)
print("y=", yf)
print("residual=",[r_prim,r_dual])
print()

xf, yf, residual = python_c_helpers.qp_solve(G=G, P=P, q=q, A=A,
                                             l=l, u=u,
                                             rho=rho, sigma=sigma, alpha=alpha,
                                             x0=x0, y0=y0, maxiter=maxiter)
print("C implementation of qp_solver, with Python's G matrix: ")
print("x=", xf)
print("y=", yf)
print("residual=", residual)

'''
xout=x0.copy().astype(np.float32)
yout=y0.copy().astype(np.float32)
residual=np.zeros(2).astype(np.float32)
c_lib.qp_solve(len(q),
               len(A),
               G_C.astype(np.float32),
               P.astype(np.float32),
               A.astype(np.float32),
               rho, sigma, alpha,
               q.astype(np.float32),
               l.astype(np.float32),
               u.astype(np.float32),
               maxiter,
               xout.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               yout.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               residual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

print("C implementation of qp_solver, with C's G matrix: ")
print("x=", xout)
print("y=", yout)
print("residual=", residual)
'''
