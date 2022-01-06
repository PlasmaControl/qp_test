import numpy as np
import matplotlib.pyplot as plt
import time
from helpers import *

'''
For a spring with mass m, spring constant k, damping b,
positive force u:
\dot{x}=Ax+Bu for 
A=[[0,1],[-k/m,-b/m]] B=[0,1/m]
(see 433 notes)
For our case of discrete time, we simply multiply 
A and B by delta_t, and add the identity to A
'''
m=1
k=1
b=0
delta_t=0.02
A=np.array([[1,delta_t],[-k/m*delta_t,1-b/m*delta_t]])
B=np.atleast_2d(np.array([0,1/m*delta_t])).T
Q=np.array([[1,0],[0,1]])
R=np.array([[1]])
Nlook=3
sigma=1.4
rho=0.1

(E_python,F_python,P_python,G_python,
 Ac_python,Qhat_python,Rhat_python)=mpc_setup(Nlook=Nlook, A=A, B=B, Q=Q, R=R, 
                                                 sigma=sigma, rho=rho, delta_t=delta_t)

alpha=1.6
epsilon=1e-4
nIter=5
z=np.array([1,0])
r=np.array([0,0])
uMin=np.array([-1])
uMax=np.array([2])
uref=np.array([0])
delta_uMin=np.array([-1])
delta_uMax=np.array([1])

nu=1
nx=2
u=np.zeros(nu*Nlook)
lamb=np.zeros(nu*(2*Nlook-1))

num_sim_timesteps=1000

X_uncontrolled = np.zeros(nx*num_sim_timesteps)
X_uncontrolled[:nx]=z
u_uncontrolled=np.zeros(nu*num_sim_timesteps)

X_MPC = np.zeros(nx*num_sim_timesteps)
X_MPC[:nx]=z
lamb=np.zeros(nu*(2*Nlook-1))
u_MPC=np.zeros(nu*num_sim_timesteps)

runtimes=[]
for i in range(1,num_sim_timesteps):
    u_uncontrolled[i*nu:(i+1)*nu] = 0
    X_uncontrolled[i*nx:(i+1)*nx] = A@X_uncontrolled[(i-1)*nx:i*nx]+B@u_uncontrolled[i*nu:(i+1)*nu]
    
    before_time=time.perf_counter()
    (u,lamb)=mpc_solve(Nlook,
                       r, z, uMin, uMax,
                       uref, delta_uMin, delta_uMax,
                       E_python, F_python, P_python, G_python, 
                       Ac_python, Qhat_python, Rhat_python,
                       rho, sigma, alpha, epsilon, nIter,
                       u,
                       lamb)
    u_MPC[i*nu:(i+1)*nu]=u[:nu]
    after_time=time.perf_counter()
    runtimes.append(after_time-before_time)
    X_MPC[i*nx:(i+1)*nx] = A@X_MPC[(i-1)*nx:i*nx]+B@u_MPC[i*nu:(i+1)*nu]

times=np.linspace(0,num_sim_timesteps*delta_t,num_sim_timesteps)
fig,axes=plt.subplots(nx+nu,sharex=True)
for i in range(nx):
    axes[i].plot(times,X_uncontrolled[i::nx],label='uncontrolled')
    axes[i].plot(times,X_MPC[i::nx],label='MPC')
for i in range(nu):
    axes[i+nx].plot(times,u_uncontrolled[i::nu],label='uncontrolled')
    axes[i+nx].plot(times,u_MPC[i::nu],label='MPC')
axes[0].set_title('States')
axes[nx].set_title('Controls')
plt.legend()
plt.show()
