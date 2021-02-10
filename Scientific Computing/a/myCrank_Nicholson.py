import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags
import sys


def xt_points(mx, mt, L, T):
    # set up the numerical environment variables

    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    return x, t

def U(func, x, L):
    # set up the solution variables
    u_j = np.zeros(x.size)    # u at current time step

    # Set initial condition
    for i in range(0, len(x)):
        u_j[i] = func(x[i], L)

    return u_j

def ThomasSolver(LHS, RHS):
    """
    A function that uses the Thomas algorithm to solve a given
    linear matrix equation.

    Parameters
    ----------
    LHS : square matrix
        Tridiagonal matrix, left hand side of the linear matrix equation
        (neglecting the vector to be solved), in the Crank-Nicholson 
        method this is the matrix A

    RHS : 1_D vector
        Right hand side of the linear matrix equation to be solved

    Returns
    -------
    Returns the vector of u_jp1 -  the solution of the linear matrix equation

    """
    a = np.zeros(LHS.shape[0])
    b = np.zeros(LHS.shape[0])
    c = np.zeros(LHS.shape[0])
    d = RHS

    for i in range(LHS.shape[0]-1):
        a[i+1] = LHS[i+1,i]
        c[i] = LHS[i,i+1]

    for i in range(LHS.shape[0]):
        b[i] = LHS[i,i]

    w = np.zeros(LHS.shape[0])
    u_jp1 = np.zeros(LHS.shape[0])

    for i in range(1,LHS.shape[0]):
        w[i] = a[i]/b[i-1]
        b[i] = b[i] - w[i]*c[i-1]
        d[i] = d[i] - w[i]*d[i-1]
        a[i] = 0
    
    u_jp1[-1] = d[-1]/b[-1]
    for i in range(LHS.shape[0]-2, -1, -1):
        u_jp1[i] = (d[i] - c[i]*u_jp1[i+1])/b[i]

    return u_jp1

def x_central_difference(mx, deltax):
    # Produces vectors of x(i + 1/2) and x(i - 1/2) over the whole x vector

    x_j_mhalf = np.zeros(mx+1)
    x_j_phalf = np.zeros(mx+1)

    # Loop over all space points
    for n in range(mx+1):

        x_jm1 = (n - 1) * deltax
        x_j = n * deltax
        x_jp1 = (n + 1) * deltax

        x_j_mhalf[n] = (x_jm1 + x_j)/2
        x_j_phalf[n] = (x_j + x_jp1)/2

    return x_j_mhalf, x_j_phalf
