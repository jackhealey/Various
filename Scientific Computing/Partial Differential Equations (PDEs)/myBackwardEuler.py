# simple backward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags
import sys

def u_I(x, L):
    # initial temperature distribution

    y = np.sin(pi*x/L)
    return y

def u_exact(x, t, kappa, L):
    # the exact solution

    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def xt_points(mx, mt, L, T):
    # set up the numerical environment variables

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    return x, t

def mf_num(mx, mt, L, T, kappa):
    '''
    USAGE:
        Insert 'p_delta' as an argument in the terminal
        to print the values of delta_x, delta_t, and lambda.

    '''
    x, t = xt_points(mx, mt, L, T)
    deltax = x[1] - x[0]                # gridspacing in x
    deltat = t[1] - t[0]                # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    if 'p_delta' in sys.argv[:]:
        print("deltax=",deltax)
        print("deltat=",deltat)
        print("lambda=",lmbda)
    return lmbda

def tridiag_A(mx, mt, L, T, kappa):
    # tridiagonal matrix for backward Euler scheme matrix form

    lmbda = mf_num(mx, mt, L, T, kappa)
    A_BE = diags([-lmbda, 1 + 2*lmbda, -lmbda], 
                        [-1, 0, 1], shape=(mx-1, mx-1)).toarray()
    return A_BE

def U(func, x, L):
    # set up the solution variables
    u_j = np.zeros(x.size)    # u at current time step

    # Set initial condition
    for i in range(0, len(x)):
        u_j[i] = func(x[i], L)

    return u_j

def ThomasSolver(A_BE, u_j):
    """
    A function that uses the Thomas algorithm to solve a given
    linear matrix equation.

    Parameters
    ----------
    A_BE : square matrix
        Tridiagonal matrix for backward Euler scheme matrix form

    u_j : 1_D vector
        Right hand side of the linear matrix equation to be solved

    Returns
    -------
    Returns the vector of u_jp1 -  the solution of the linear matrix equation

    """
    a = np.zeros(len(u_j)) + A_BE[1,0]; a[0] = 0
    b = np.zeros(len(u_j)) + A_BE[1,1]
    c = np.zeros(len(u_j)) + A_BE[1,2]; c[-1] = 0
    d = u_j

    w = np.zeros(len(u_j))
    u_jp1 = np.zeros(len(u_j))

    for i in range(1,len(u_j)):
        w[i] = a[i]/b[i-1]
        b[i] = b[i] - w[i]*c[i-1]
        d[i] = d[i] - w[i]*d[i-1]
        a[i] = 0
    
    u_jp1[-1] = d[-1]/b[-1]
    for i in range(len(u_j)-2, -1, -1):
        u_jp1[i] = (d[i] - c[i]*u_jp1[i+1])/b[i]

    return u_jp1

def Backward_Euler_solver(func, mx, mt, L, T, kappa, u_0, u_T, bCond):
    """
    A function that uses the backward Euler scheme in matrix form
    to solve a given function.

    Parameters
    ----------
    func : function of initial temperature distribution
        Function at u(x,0), T = 0

    mx : integer
        The number of gridpoints in space.

    mt : integer
        The number of gridpoints in time.

    L : parameter
        Length of spatial domain
    
    T : parameter
        Total time to solve for

    kappa : parameter
        Diffusion constant

    u_0 , u_T : Dirichlet boundary conditions

    Returns
    -------
    Returns the mesh points in space, along with a numpy.ndarray
    of the solutions.

    """
    x,_ = xt_points(mx, mt, L, T)
    u_j = U(func, x, L)
    u_jp1 = np.zeros(len(u_j))
    A_BE = tridiag_A(mx, mt, L, T, kappa)

    # Solve the PDE: loop over all time points
    for n in range(1, mt+1):
        # Backward Euler scheme in matrix form at inner mesh points
            #u_jp1 = np.linalg.solve(A_BE, u_j)
        u_jp1[1:-1] = ThomasSolver(A_BE, u_j[1:-1])
        # Boundary conditions
        u_jp1[0] = u_0; u_jp1[-1] = u_T
        # Update u_j
        u_j = u_jp1

    return x, u_j

x, u_j = Backward_Euler_solver(
                                func = u_I,
                                mx = 20,
                                mt = 1000,
                                L = 1.0,
                                T = 0.5,
                                kappa = 1.0,
                                u_0 = 0,
                                u_T = 0,
                                bCond = 'D_hom'
                                )

# # plot the final result and exact solution
# pl.plot(x,u_j,'ro',label='num')

# L = 1.0; T = 0.5; xx = np.linspace(0,L,250);

# pl.title('Backwards Euler')
# pl.plot(xx,u_exact(xx,T, kappa = 1.0, L = 1.0),'b-',label='exact')
# pl.xlabel('x')
# pl.ylabel('u(x,{})'.format(T))
# pl.legend(loc='upper right')
# pl.show()