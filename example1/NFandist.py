import numpy as np
from NFconstants import N_nod, Beta

def get_O(N):
    O=np.zeros((N,N))
    for k in range(N):
        O[k][0]=1
    
    if N%2==0:
        for k in range(N):
            O[k][1]=(-1)**k
    
    ev=int(N%2==0)
    Smax=(N-1-ev)//2    
    
    for s in range(1,Smax+1):
        for k in range(N):
            O[k][2*s-1+ev]=(2**0.5)*np.cos(2*np.pi*s*k/N)
            O[k][2*s+ev]=(2**0.5)*np.sin(2*np.pi*s*k/N)

    O=O / (N ** 0.5)
    return O

def Lambda(k,N,beta):
    a = beta / N
    L = a + ( 4 / a ) * (np.sin( np.pi * k / N )) ** 2 
    return L

def get_diag(N,beta):
    a = beta/N
    n = [0]*N
    ev = int(N%2==0)
    Smax = (N-1-ev)//2 
    
    n[0]=(Lambda(0,N,beta)) ** -0.5
    
    if ev:
        n[1] = (Lambda(N//2,N,beta)) ** -0.5
        
    for s in range(1,Smax+1):
        n[2*s-1+ev] = (Lambda(s,N,beta)) ** -0.5
        n[2*s+ev] = (Lambda(s,N,beta)) ** -0.5
    return n    

def get_diag_mat(N,beta):
    n=get_diag(N,beta)
    diag_mat=np.diag(n)
    return diag_mat

def get_C(N,beta):
    O=get_O(N)
    diag_mat=get_diag_mat(N,beta)
    C=np.dot(O,diag_mat)
    return C

def get_A(N,beta):
    A=np.zeros((N,N))
    a=beta/N
    for i in range(N_nod):
        A[i][(i-1)%N_nod]=-1
        A[i][i]=2+a**2
        A[i][(i+1)%N_nod]=-1
    A=A/a   
    return A

def get_T(N):
    T=np.eye(N)
    T=np.roll(T,1,axis=1)
    return T

def get_Ainv(N,beta):
    C=get_C(N,beta)
    Ct=np.transpose(C)
    A_inv=np.dot(C,Ct)
    return A_inv

def calc_G(N,beta,n_points="all"):
    if n_points=="all":
        n_points=N
    
    G=np.zeros((n_points))
    T=get_T(N)
    g=get_Ainv(N,beta)
    for i in range(n_points):
        G[i]=np.trace(g)
        g=np.dot(g,T)
    return G/N    

def calc_sigmaG(N,beta,n_points="all"):
    if n_points=="all":
        n_points=N
    
    dispG=np.zeros((n_points))
    T=get_T(N)
    g=get_Ainv(N,beta)
    for i in range(n_points):
        dispG[i]=np.trace(np.dot(g,g))
        g=np.dot(g,T)
    dispG=(2 / N ** 2) * dispG
    sigmaG=dispG ** 0.5
    return sigmaG 

def calc_Z(N,beta):
    k = np.arange(N)
    L = Lambda(k,N,beta)
    a = beta / N
    L = a * L
    prod=np.prod(L)
    Z=(prod) ** -0.5
    return Z

def Z_inf(beta):
    return 1 / (2 * np.sinh(beta//2) )

def calc_Kin(N,beta):
    a=beta/N
    g=calc_G(N,beta,2)
    K = (1 - 2 * (g[0]-g[1]) / a) / (2 * a)
    return K
    