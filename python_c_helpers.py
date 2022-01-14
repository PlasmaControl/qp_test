import ctypes
import numpy as np

def array_to_ctypes_1d(A):
    # initialize properly sized array (to 0)
    arr=(ctypes.c_float*len(A))()
    for i in range(len(A)):
        arr[i]=A[i]
    return arr

def array_to_ctypes_2d(A):
    # initialize properly sized array (to 0)
    arr=((ctypes.c_float*len(A[0]))*len(A))()
    for i in range(len(A)):
        for j in range(len(A[0])):
            arr[i][j]=A[i][j]
    return arr

def ctypes_to_array_1d(A,N):
    arr=np.zeros((N))
    for i in range(N):
        arr[i]=A[i]
    return arr

# A an NxM matrix
def ctypes_to_array_2d(A,N,M):
    arr=np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            arr[i][j]=A[i][j]
    return arr

def qp_setup(P,A,rho,sigma):
    N=A.shape[1]
    M=A.shape[0]

    clib=ctypes.cdll.LoadLibrary('./libqp_solver.so')
    clib.qp_setup.restype=None
    clib.qp_setup.argtypes=[ctypes.c_size_t, #N
                            ctypes.c_size_t, #M
                            (ctypes.c_float*N)*N, #P
                            (ctypes.c_float*N)*M, #A
                            ctypes.c_float, #sigma
                            ctypes.c_float, #rho
                            (ctypes.c_float*N)*N] #G

    P=array_to_ctypes_2d(P)
    A=array_to_ctypes_2d(A)
    G=((ctypes.c_float*N)*N)()

    clib.qp_setup(N, M,
                  P, A, 
                  sigma, rho, 
                  G)
    return ctypes_to_array_2d(G,N,N)

def qp_solve(G, P, q, A, l, u, rho, sigma, alpha, x0, y0, maxiter):
    N=A.shape[1]
    M=A.shape[0]

    clib=ctypes.cdll.LoadLibrary('./libqp_solver.so')
    clib.qp_solve.restype=None
    clib.qp_solve.argtypes=[ctypes.c_size_t, #N
                            ctypes.c_size_t, #M
                            (ctypes.c_float*N)*N, #G
                            (ctypes.c_float*N)*N, #P
                            (ctypes.c_float*N)*M, #A
                            ctypes.c_float, #rho
                            ctypes.c_float, #sigma
                            ctypes.c_float, #alpha
                            ctypes.c_float*N, #q
                            ctypes.c_float*M, #l
                            ctypes.c_float*M, #u
                            ctypes.c_size_t,
                            ctypes.c_float*N, #xOut
                            ctypes.c_float*M, #yOut
                            ctypes.c_float*2] #residual

    G=array_to_ctypes_2d(G)
    P=array_to_ctypes_2d(P)
    A=array_to_ctypes_2d(A)
    q=array_to_ctypes_1d(q)
    l=array_to_ctypes_1d(l)
    u=array_to_ctypes_1d(u)
    x0=array_to_ctypes_1d(x0)
    y0=array_to_ctypes_1d(y0)
    residuals=(ctypes.c_float*2)()

    clib.qp_solve(N, M,
                  G, P, A,
                  rho, sigma, alpha,
                  q, l, u, maxiter,
                  x0, y0, residuals)
    return (ctypes_to_array_1d(x0,N),
            ctypes_to_array_1d(y0,M),
            ctypes_to_array_1d(residuals,2))

def mpc_setup(A, B, Q, R, Nlook, rho, sigma, dt, use_osqp=False):
    nu=B.shape[1]
    nz=B.shape[0]

    clib=ctypes.cdll.LoadLibrary('./libqp_solver.so')
    clib.mpc_setup.restype=None
    clib.mpc_setup.argtypes=[ctypes.c_size_t, #nZ
                            ctypes.c_size_t, #nU
                            ctypes.c_size_t, #nLook
                            (ctypes.c_float*nz)*nz, #A
                            (ctypes.c_float*nu)*nz, #B
                            (ctypes.c_float*nz)*nz, #Q
                            (ctypes.c_float*nu)*nu, #R
                            ctypes.c_float, #sigma
                            ctypes.c_float, #rho
                            ctypes.c_float, #deltaT
                            (ctypes.c_float*nz)*(nz*Nlook), #E
                            (ctypes.c_float*(nu*Nlook))*(nz*Nlook), #F
                            (ctypes.c_float*(nu*Nlook))*(nu*Nlook), #P
                            (ctypes.c_float*(nu*Nlook))*(nu*Nlook), #G
                            (ctypes.c_float*(nu*Nlook))*(nu*(2*Nlook-1)), #Ac
                            ctypes.c_float*(nz*Nlook), #Qhat
                            ctypes.c_float*(nu*Nlook)] #Rhat

    A=array_to_ctypes_2d(A)
    B=array_to_ctypes_2d(B)
    Q=array_to_ctypes_2d(Q)
    R=array_to_ctypes_2d(R)

    E=((ctypes.c_float*nz)*(nz*Nlook))()
    F=((ctypes.c_float*(nu*Nlook))*(nz*Nlook))()
    P=((ctypes.c_float*(nu*Nlook))*(nu*Nlook))()
    G=((ctypes.c_float*(nu*Nlook))*(nu*Nlook))()
    Ac=((ctypes.c_float*(nu*Nlook))*(nu*(2*Nlook-1)))()
    Qhat=(ctypes.c_float*(nz*Nlook))()
    Rhat=(ctypes.c_float*(nu*Nlook))()

    clib.mpc_setup(nz, nu, Nlook,
                   A, B, Q, R, 
                   sigma, rho, dt,
                   E, F, P, G, Ac, Qhat, Rhat)
    return (ctypes_to_array_2d(E,nz*Nlook,nz),
            ctypes_to_array_2d(F,nz*Nlook,nu*Nlook),
            ctypes_to_array_2d(P,nu*Nlook,nu*Nlook),
            ctypes_to_array_2d(G,nu*Nlook,nu*Nlook),
            ctypes_to_array_2d(Ac,nu*(2*Nlook-1),nu*Nlook),
            np.diag(ctypes_to_array_1d(Qhat,nz*Nlook)),
            np.diag(ctypes_to_array_1d(Rhat,nu*Nlook)))

def mpc_action(
    zk,
    ztarget,
    uhat,
    lagrange,
    uminhat,
    duminhat,
    urefhat,
    dumaxhat,
    umaxhat,
    E,
    F,
    P,
    G,
    Ac,
    Qhat,
    Rhat,
    rho,
    sigma,
    alpha,
    maxiter,
    qp=None,
):
    nz=len(zk)
    Nlook=int(len(Qhat)/nz)
    nu=int(len(Rhat)/Nlook)

    clib=ctypes.cdll.LoadLibrary('./libqp_solver.so')
    clib.mpc_solve.restype=None
    clib.mpc_solve.argtypes=[ctypes.c_size_t, #nZ
                             ctypes.c_size_t, #nU
                             ctypes.c_size_t, #nLook
                             ctypes.c_size_t, #nIter
                             ctypes.c_float, #rho
                             ctypes.c_float, #sigma
                             ctypes.c_float, #alpha
                             ctypes.c_float*nz, #z
                             ctypes.c_float*(nz*Nlook), #rhat
                             ctypes.c_float*(nu*Nlook), #uhatmin
                             ctypes.c_float*(nu*Nlook), #uhatmax
                             ctypes.c_float*(nu*(Nlook-1)), #uhatmindelta
                             ctypes.c_float*(nu*(Nlook-1)), #uhatmaxdelta
                             ctypes.c_float*(nu*Nlook), #uhatref
                             (ctypes.c_float*nz)*(nz*Nlook), #E
                             (ctypes.c_float*(nu*Nlook))*(nz*Nlook), #F
                             (ctypes.c_float*(nu*Nlook))*(nu*Nlook), #P
                             (ctypes.c_float*(nu*Nlook))*(nu*Nlook), #G
                             (ctypes.c_float*(nu*Nlook))*(nu*(2*Nlook-1)), #Ac
                             ctypes.c_float*(nz*Nlook), #Qhat
                             ctypes.c_float*(nu*Nlook), #Rhat
                             ctypes.c_float*(nu*Nlook), #uHat
                             ctypes.c_float*(nu*(2*Nlook-1))] #lambda

    ztarget=array_to_ctypes_1d(ztarget)
    zk=array_to_ctypes_1d(zk)
    uminhat=array_to_ctypes_1d(uminhat)
    umaxhat=array_to_ctypes_1d(umaxhat)
    urefhat=array_to_ctypes_1d(urefhat)
    duminhat=array_to_ctypes_1d(duminhat)
    dumaxhat=array_to_ctypes_1d(dumaxhat)
    E=array_to_ctypes_2d(E)
    F=array_to_ctypes_2d(F)
    P=array_to_ctypes_2d(P)
    G=array_to_ctypes_2d(G)
    Ac=array_to_ctypes_2d(Ac)
    Qhat=array_to_ctypes_1d(np.diag(Qhat))
    Rhat=array_to_ctypes_1d(np.diag(Rhat))

    uhat=array_to_ctypes_1d(uhat)
    lagrange=array_to_ctypes_1d(lagrange)

    clib.mpc_solve(nz, nu, Nlook,
                   maxiter, 
                   rho, sigma, alpha,
                   zk,
                   ztarget,
                   uminhat, umaxhat, duminhat, dumaxhat,
                   urefhat, 
                   E, F, P, G, Ac, Qhat, Rhat,
                   uhat, lagrange)
    return (ctypes_to_array_1d(uhat,nu*Nlook),
            ctypes_to_array_1d(lagrange,nu*(2*Nlook-1)))

if __name__ == "__main__":
    n = 5 # number of variables
    m = 4 # number of constraints

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

    sigma = 1e-6
    rho = 6
    alpha=1.6
    maxiter=3

    x0 = np.zeros(n)
    y0 = np.zeros(m)
    
    print("compare to qp_solver.c")

    
    G=qp_setup(P,A,rho,sigma)
    print("G: ")
    print(G)
    (xf, yf, residuals)=qp_solve(G,P,A,rho,sigma,alpha,
                                 q,l,u,x0,y0,maxiter)
    print("x0:")
    print(xf)
    print("y0:")
    print(yf)
    print("residuals:")
    print(residuals)


    
