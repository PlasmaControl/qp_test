# https://realpython.com/python-bindings-overview/
import ctypes
import os
import numpy as np
import osqp
import scipy
import scipy.linalg
import time
import matplotlib.pyplot as plt

# set up to call C function vep_box from python
libname=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "libqp_solver.so")
c_lib=ctypes.CDLL(libname)

c_lib.vep_box.argtypes = [
    ctypes.c_float, #lambda
    ctypes.c_size_t, #N
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #phi
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), #Gamma
    # we want to change the value of an input array, so have to 
    # convert to a pointer (C passes by value, so we need to supply
    # an address to copy rather than an array of data)
    ctypes.POINTER(ctypes.c_double), #xOut
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), #l
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')] #u
c_lib.vep_box.restype = ctypes.c_void_p #void return

### TEST 1 ###
'''
lamb=1
N=2
phi=np.zeros(N).astype(np.float32)
gamma=np.identity(N).astype(np.float32)
xout=np.array([2,.3]).astype(np.float32)
l=-1*np.ones(N).astype(np.float32)
u=1*np.ones(N).astype(np.float32)
answer=[0,0]
'''
### TEST 2 ###
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
Q=[[1,0],[0,1]]
R=[[1]]

#LQR
P=scipy.linalg.solve_discrete_are(A,B,Q,R)
K=scipy.linalg.inv(R) @ B.T @ P

# MPC pre-computation
horizon=1
nx=np.shape(B)[0]
nu=np.shape(B)[1]
x0=np.array([0,0.01]).T
E=np.zeros((horizon*nx,nx))
F=np.zeros((horizon*nx,horizon*nu))
E[:nx]=A
for i in range(horizon):
    F[i*nx:(i+1)*nx,i*nu:(i+1)*nu]=B
for i in range(1,horizon):
    E[i*nx:(i+1)*nx]=E[(i-1)*nx:i*nx] @ A
    for j in range(i):
        F[i*nx:(i+1)*nx,j*nu:(j+1)*nu]=E[j*nx:(j+1)*nx] @ B
Qhat=[Q]*horizon
Rhat=[R]*horizon
Qhat[-1]=P #LQR to cut off the time
Qhat=scipy.linalg.block_diag(*Qhat)
Rhat=scipy.linalg.block_diag(*Rhat)
phi=np.ones(nu*horizon)
gamma=F.T@Qhat@F+Rhat
gamma=(gamma+gamma.T) # ensure exactly symmetric and add factor of 2
qp=osqp.OSQP()
Aineq=np.eye(horizon*nu)
l=-1000*np.ones(nu*horizon)
u=1000*np.ones(nu*horizon)
qp.setup(P=scipy.sparse.csc_matrix(gamma),
         q=phi,
         A=scipy.sparse.csc_matrix(Aineq),
         l=l, u=u, 
         verbose=False)
def get_u_mpc(x):
    phi=2*x.T@E.T@Qhat@F
    qp.update(q=phi)
    results=qp.solve()
    return results.x[:nu]
def get_u_mpc_pcs(x):
    phi=2*x.T@E.T@Qhat@F
    xout=np.ones(nu*horizon)

    # we want to change the value of an input array, so have to 
    # convert to a pointer (C passes by value, so we need to supply
    # an address to copy rather than an array of data)
    xout=xout.astype(np.float32)
    c_lib.vep_box(1, #lambda
                  nu*horizon, 
                  phi.astype(np.float32), 
                  gamma.astype(np.float32), 
                  xout.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  l.astype(np.float32), 
                  u.astype(np.float32))
    return xout[:nu]

num_sim_timesteps=1000

X_uncontrolled = np.zeros(nx*num_sim_timesteps)
X_uncontrolled[:nx]=x0
u_uncontrolled=np.zeros(nu*num_sim_timesteps)

X_LQR = np.zeros(nx*num_sim_timesteps)
X_LQR[:nx]=x0
u_LQR=np.zeros(nu*num_sim_timesteps)

X_MPC = np.zeros(nx*num_sim_timesteps)
X_MPC[:nx]=x0
u_MPC=np.zeros(nu*num_sim_timesteps)

runtimes=[]
for i in range(1,num_sim_timesteps):
    u_uncontrolled[i*nu:(i+1)*nu] = 0
    X_uncontrolled[i*nx:(i+1)*nx] = A@X_uncontrolled[(i-1)*nx:i*nx]+B@u_uncontrolled[i*nu:(i+1)*nu]

    u_LQR[i*nu:(i+1)*nu]=-K@X_LQR[(i-1)*nx:i*nx]
    X_LQR[i*nx:(i+1)*nx] = A@X_LQR[(i-1)*nx:i*nx]+B@u_LQR[i*nu:(i+1)*nu]

    before_time=time.perf_counter()
    u_MPC[i*nu:(i+1)*nu] = get_u_mpc_pcs(X_MPC[(i-1)*nx:i*nx])
    after_time=time.perf_counter()
    runtimes.append(after_time-before_time)
    X_MPC[i*nx:(i+1)*nx] = A@X_MPC[(i-1)*nx:i*nx]+B@u_MPC[i*nu:(i+1)*nu]
print(f'Horizon={horizon}: Took {np.mean(runtimes):.2e}+/-{np.std(runtimes):.2e}')
'''
x_target=np.array([0,0]).T
Q=np.eye(nx)
R=np.eye(nu)
U=scipy.linalg.solve_discrete_are(A,B,Q,R)
X_LQR = E@x0 + F@U
'''

if True:
    times=np.linspace(0,num_sim_timesteps*deltaT,num_sim_timesteps)
    fig,axes=plt.subplots(nx+nu)
    for i in range(nx):
        axes[i].plot(times,X_uncontrolled[i::nx],label='uncontrolled')
        axes[i].plot(times,X_LQR[i::nx],label='LQR')
        axes[i].plot(times,X_MPC[i::nx],label='MPC')
    for i in range(nu):
        axes[i+nx].plot(times,u_uncontrolled[i::nu],label='uncontrolled')
        axes[i+nx].plot(times,u_LQR[i::nu],label='LQR')
        axes[i+nx].plot(times,u_MPC[i::nu],label='MPC')
    axes[0].set_title('States')
    axes[nx].set_title('Controls')
    plt.legend()
    plt.show()
'''

lamb=1
N=2
phi=np.zeros(N).astype(np.float32)
gamma=np.identity(N).astype(np.float32)
xout=np.array([2,.3]).astype(np.float32)
l=-1*np.ones(N).astype(np.float32)
u=1*np.ones(N).astype(np.float32)
answer=[0,0]



tic=time.perf_counter()
c_lib.vep_box(lamb, N, phi, gamma, xout, l, u)
toc=time.perf_counter()

print()
print(f'Should be {answer}: ')
print()
print(f'PCS: {np.all(np.isclose(answer,xout))}')
#print(xout)
print(f'Took {toc-tic:0.4f}s')
print()

qp = osqp.OSQP()
qp.setup(P=scipy.sparse.csc_matrix(gamma), 
         q=phi, 
         A=scipy.sparse.csc_matrix(np.eye(N)), 
         l=l, u=u, 
         verbose=False)
tic=time.perf_counter()
results=qp.solve()
toc=time.perf_counter()
print(f'OSQP: {np.all(np.isclose(answer,results.x))}')
#print(results.x)
print(f'Took {toc-tic:0.4f}s')
'''
