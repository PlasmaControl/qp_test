import numpy as np
from helpers import *

'''
For a spring with mass m, spring constant k, damping b,
positive force u:
\dot{x}=Ax+Bu for 
A=[[0,1],[-k/m,-b/m]] B=[0,1/m]
(see 433 notes)
For our case of discrete time, we simply multiply 
A and B by deltaT, and add the identity to A
'''
m=1
k=1
b=0
deltaT=0.02
A=np.array([[1,deltaT],[-k/m*deltaT,1-b/m*deltaT]])
B=np.atleast_2d(np.array([0,1/m*deltaT])).T
Q=np.array([[1,0],[0,1]])
R=np.array([[1]])
Nlook=3

sigma=1
rho=1
delta_t=2

import ctypes
import os
libname=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "libqp_solver.so")
c_lib=ctypes.CDLL(libname)

c_lib.mpc_setup.argtypes = [
    ctypes.c_size_t, #nZ
    ctypes.c_size_t, #nU
    ctypes.c_size_t, #nLook
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #A
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #B
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #Q
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #R
    ctypes.c_float, #sigma
    ctypes.c_float, #rho
    ctypes.c_float, #delta_t
    # we want to change the value of an input array, so have to 
    # convert to a pointer (C passes by value, so we need to supply
    # an address to copy rather than an array of data)
    ctypes.POINTER(ctypes.c_double), #E
    ctypes.POINTER(ctypes.c_double), #F
    ctypes.POINTER(ctypes.c_double), #P
    ctypes.POINTER(ctypes.c_double), #G
    ctypes.POINTER(ctypes.c_double), #Ac
    ctypes.POINTER(ctypes.c_double), #Qhat
    ctypes.POINTER(ctypes.c_double)] #Rhat
c_lib.qp_solve.restype = ctypes.c_void_p #void return (answer goes into xOut)

nz=len(Q)
nu=len(R)
E=np.zeros((nz,nz*Nlook)).astype(np.float32)
F=np.zeros((nz*Nlook,nu*Nlook)).astype(np.float32)
P=np.zeros((nu*Nlook,nu*Nlook)).astype(np.float32)
G=np.zeros((nu*Nlook,nu*Nlook)).astype(np.float32)
Ac=np.zeros((nu*(2*Nlook-1),nu*Nlook)).astype(np.float32)
Qhat=np.zeros((nz*Nlook)).astype(np.float32)
Rhat=np.zeros((nu*Nlook)).astype(np.float32)
c_lib.mpc_setup(nz,
                nu,
                Nlook,
                A.astype(np.float32),
                B.astype(np.float32),
                Q.astype(np.float32),
                R.astype(np.float32),
                sigma, rho, delta_t,
                E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                F.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                P.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                G.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                Ac.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                Qhat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                Rhat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

(E_python,F_python,P_python,
 Ac_python,Qhat_python,Rhat_python)=mpc_setup(Nlook=Nlook, A=A, B=B, Q=Q, R=R, 
                                                 sigma=sigma, rho=rho, delta_t=delta_t)

print("E")
print("Python implementation:")
print(E_python)
print("C implementation")
print(E)
print()

print("F")
print("Python implementation:")
print(F_python)
print("C implementation")
print(F)
print()

print("P")
print("Python implementation:")
print(P_python)
print("C implementation")
print(P)
print()

print("Ac")
print("Python implementation:")
print(Ac_python)
print("C implementation")
print(Ac)
print()

print("Qhat")
print("Python implementation:")
print(Qhat_python)
print("C implementation")
print(Qhat)
print()

print("Rhat")
print("Python implementation:")
print(Rhat_python)
print("C implementation")
print(Rhat)
