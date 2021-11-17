# https://realpython.com/python-bindings-overview/
import ctypes
import os
import numpy as np

libname=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "libqp_solver.so")
c_lib=ctypes.CDLL(libname)

c_lib.vep_box.argtypes = [
    ctypes.c_float, #lambda
    ctypes.c_size_t, #N
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #phi
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #Gamma
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #xOut
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #l
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')] #u
c_lib.vep_box.restype = ctypes.c_void_p #void return

python_lamb=1
python_N=2
python_phi=np.zeros(python_N).astype(np.float32)
python_gamma=np.identity(python_N).astype(np.float32)
python_xout=np.array([2,.3]).astype(np.float32)
python_l=-1*np.ones(python_N).astype(np.float32)
python_u=1*np.ones(python_N).astype(np.float32)

c_lib.vep_box(python_lamb, python_N, python_phi, python_gamma, python_xout, python_l, python_u)

print(python_xout)
