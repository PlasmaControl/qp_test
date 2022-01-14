import numpy as np
import mpc_qp_helpers
import python_c_helpers

if True:
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
    nu=1
    nz=2

    alpha=1.6
    nIter=3
    uref=np.array([0])
    xref=np.array([0,0]) #target position 0, velocity 0
    umin=np.array([-1])
    umax=np.array([1])
    delta_umin=np.array([-1])
    delta_umax=np.array([1])

    x0=np.array([0,0.01]) #initial position 0, velocity 0

else:
    '''
    For 2 carts with independent engines, wall on left to spring to first
    cart to second spring to second cart. x-l1 the first variable, y-l2
    the second for l1 and l2 the relaxed spring lengths. 
    '''
    m1=1
    m2=1
    k1=1
    k2=1
    b1=0.1
    b2=0.1
    delta_t=0.02
    nu=2
    nx=4
    A=np.array([[1,              delta_t,         0,              0],
                [-k1/m1*delta_t, 1-b1/m1*delta_t, 0,              0],
                [0,              0,               1,              delta_t],
                [k2/m2*delta_t,  0,               -k2/m2*delta_t, 1-b2/m2*delta_t]])
    B=np.array([[0,            0],
                [1/m1*delta_t, 0],
                [0,            0],
                [0,            1/m2*delta_t]])
    Q=np.eye(nx)
    R=np.eye(nu)
    Nlook=100
    sigma=1e-4 #1.4
    rho=0.1

    alpha=1.6
    nIter=5
    uref=np.array([0,0])
    xref=np.array([0,0,0,0]) #target position 0, velocity 0
    umin=np.array([-np.inf,-np.inf])
    umax=np.array([np.inf,np.inf])
    delta_umin=np.array([-1,-1])
    delta_umax=np.array([1,1])

    x0=np.array([0.01,0,0,0]) 

Nlook=3
sigma=1.4
rho=0.1

(E,F,P,G,
 Ac,Qhat,Rhat)=python_c_helpers.mpc_setup(Nlook=Nlook, A=A, B=B, Q=Q, R=R,
                                          sigma=sigma, rho=rho, dt=delta_t)

(E_python,F_python,P_python,G_python,
 Ac_python,Qhat_python,Rhat_python)=mpc_qp_helpers.mpc_setup(Nlook=Nlook, A=A, B=B, Q=Q, R=R,
                                                             sigma=sigma, rho=rho, dt=delta_t)

print("Testing mpc_setup")
print()

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


print()
print()
print('Testing mpc solve:')

u0=np.zeros((nu*Nlook)) #initial guess for best control all 0
lamb0=np.zeros((nu*(2*Nlook-1))) #initial guess for Lagrangian forces 0

uref_hat=np.tile(uref, (Nlook,))
xref_hat = np.tile(xref, (Nlook,))
umin_hat = np.tile(umin, (Nlook,))
umax_hat = np.tile(umax, (Nlook,))
delta_umin_hat = np.tile(delta_umin, (Nlook-1,))
delta_umax_hat = np.tile(delta_umax, (Nlook-1,))

(xf,yf)=mpc_qp_helpers.mpc_action(zk=x0, ztarget=xref_hat,
                                  uhat=u0, lagrange=lamb0,
                                  uminhat=umin_hat, duminhat=delta_umin_hat,
                                  umaxhat=umax_hat, dumaxhat=delta_umax_hat,
                                  urefhat=uref_hat,
                                  E=E_python, F=F_python, P=P_python,
                                  G=G_python, Ac=Ac_python,
                                  Qhat=Qhat_python, Rhat=Rhat_python,
                                  rho=rho, sigma=sigma, alpha=alpha,
                                  maxiter=nIter)
print('Python MPC solution:')
print(f'xf: {xf}')
print(f'yf: {yf}')

(xf,yf)=python_c_helpers.mpc_action(zk=x0, ztarget=xref_hat,
                                    uhat=u0, lagrange=lamb0,
                                    uminhat=umin_hat, duminhat=delta_umin_hat,
                                    umaxhat=umax_hat, dumaxhat=delta_umax_hat,
                                    urefhat=uref_hat,
                                    E=E_python, F=F_python, P=P_python,
                                    G=G_python, Ac=Ac_python,
                                    Qhat=Qhat_python, Rhat=Rhat_python,
                                    rho=rho, sigma=sigma, alpha=alpha,
                                    maxiter=nIter)
print('C MPC solution:')
print(f'xf: {xf}')
print(f'yf: {yf}')
