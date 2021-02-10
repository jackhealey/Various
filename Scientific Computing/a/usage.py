from myCrank_Nicholson import *

def u_I(x, L):
    # USER INPUT: Initial temperature distribution
    
    y = np.sin(pi*x/L)
    return y

def u_exact(x, t, kappa, L):
    # USER INPUT: The exact solution

    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def F(x, t_j):
    """
    A function for a HEAT SOURCE inside the domain, returning the result
    of the function at a time point t_j across gridpoints x in space.

    Returns F_j with input t_j and F_jp1 with input t_jp1

    Parameters
    ----------
    x : array
        Vector of gridpoints in space.

    t_j : scalar
        Time point t_j.


    USAGE - USER INPUT
    ------------------
    If there is no heat source:
        y[i] = 0

    Otherwise:
        y[i] for the left hand side
        x[i] on the right hand side when x is referenced in the equation
        t_j on the right hand side when t is referenced in the equation

        Example of syntax: y[i] = np.exp(t_j) * np.sin(x[i])

    """
    y = np.zeros(len(x))

    for i in range(len(x)):
        # heat source equation
        y[i] = 0

    return y

def kappa_func(x_i):
    """
    A function for a variable diffusion coefficient, returning the result
    of the function at a time point t_j across gridpoints x in space.

    Returns k_i with input x_i and k_i_mhalf with input x_i_mhalf

    Parameters
    ----------
    x_i : scalar
        Space point x_i.


    USAGE - USER INPUT
    ------------------
    If there is no variable diffusion coefficient:
        y = kappa = 1.0

    Otherwise:
        y for the left hand side
        x_i on the right hand side when x is referenced in the equation

        Example of syntax: y[i] = np.sin(x_i)

    """

    y = 1.0

    return y

def lmbda_calc(x, deltat, deltax):
    kappa_list = np.zeros(len(x))

    # creating kappa vector over space points
    for i in range(len(x)):
        kappa_list[i] = kappa_func(x[i])

    # calculate lambda
    lmbda = kappa_list*deltat/(deltax**2)

    return lmbda

def tridiag_kappa(mx, deltat, deltax):
    # tridiagonal matrices for Crank-Nicholson scheme matrix form, taking 
    # into account the variable diffusion coefficient kappa(x)

    A_CN = np.zeros((mx+1,  mx+1))
    B_CN = np.zeros((mx+1,  mx+1))

    x_i_mhalf, x_i_phalf = x_central_difference(mx, deltax)

    for i in range(0,mx+1):
        A_CN[i,i]   =   1 + 0.5 * (deltat/deltax**2) \
                                * (kappa_func(x_i_phalf[i]) + kappa_func(x_i_mhalf[i]))
        B_CN[i,i]   =   1 - 0.5 * (deltat/deltax**2) \
                                * (kappa_func(x_i_phalf[i]) + kappa_func(x_i_mhalf[i]))

    for i in range(0,mx):
        A_CN[i+1,i] = - 0.5 * (deltat/deltax**2) * kappa_func(x_i_mhalf[i+1])
        A_CN[i,i+1] = - 0.5 * (deltat/deltax**2) * kappa_func(x_i_mhalf[i])
        B_CN[i+1,i] =   0.5 * (deltat/deltax**2) * kappa_func(x_i_mhalf[i+1])
        B_CN[i,i+1] =   0.5 * (deltat/deltax**2) * kappa_func(x_i_mhalf[i])

    return A_CN, B_CN

def Crank_Nicholson_solver(func, mx, mt, L, T, kappa, u_0, u_T, bCond):
    """
    A function that uses the Crank-Nicholson scheme in matrix form
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

    u_0 , u_T : Boundary conditions

        Dirichlet:
                    homogenous: u_0 = 0, u_T = 0
                    non-homogenous : u_0 = lambda T : , u_T = lambda T :

        Neumann:
                    u_0 = lambda T : , u_T = lambda T :

    bCond : Label for Boundary conditions

            Dirichlet: 'D_hom', 'D_non_hom'

            Neumann: 'Neum'


    Returns
    -------
    Returns the mesh points in space, along with a numpy.ndarray
    of the solutions.

    """
    x,_ = xt_points(mx, mt, L, T)
    deltat = T/mt
    deltax = L/mx
    u_j = U(func, x, L)
    u_jp1 = np.zeros(len(u_j))
    lmbda = lmbda_calc(x, deltat, deltax)

    # Solve the PDE: loop over all time points
    for n in range(1, mt+1): 

        t_j = n * deltat
        t_jp1 = (n + 1) * deltat

        heat_source = (deltat/2)*(F(x, t_j) + F(x, t_jp1))

        A_CN, B_CN = tridiag_kappa(mx, deltat, deltax)

        if bCond == 'D_hom':

            A_CN = A_CN[1:-1,1:-1] ; B_CN = B_CN[1:-1,1:-1]

            RHS = B_CN.dot(u_j[1:-1]) + heat_source[1:-1]
            
            u_jp1[1:-1] = ThomasSolver(A_CN, RHS)
            # Boundary conditions
            u_jp1[0] = u_0; u_jp1[-1] = u_T

        elif bCond == 'D_non_hom':

            A_CN = A_CN[1:-1,1:-1] ; B_CN = B_CN[1:-1,1:-1]
            
            p_q = np.zeros(mx-1)
            p_q[0] = u_0(t_j) + u_0(t_jp1)
            p_q[-1] = u_T(t_j) + u_T(t_jp1)

            RHS = B_CN.dot(u_j[1:-1]) + (lmbda/2)*p_q + heat_source[1:-1]

            u_jp1[1:-1] = ThomasSolver(A_CN, RHS)
            # Boundary conditions
            u_jp1[0] = u_0(t_j); u_jp1[-1] = u_T(t_j)

        elif bCond == 'Neum':

            A_CN[0,1] = 2*A_CN[0,1]; A_CN[-1,-2] = 2*A_CN[-1,-2]
            B_CN[0,1] = 2*B_CN[0,1]; B_CN[-1,-2] = 2*B_CN[-1,-2]
            
            P_Q = np.zeros(mx+1)
            P_Q[0] = - u_0(t_j) - u_0(t_jp1)
            P_Q[-1] = u_T(t_j) + u_T(t_jp1)

            RHS = B_CN.dot(u_j) + lmbda*deltax*P_Q + heat_source

            u_jp1 = ThomasSolver(A_CN, RHS)

        # Update u_j
        u_j = u_jp1

    return x, u_j

x, u_j = Crank_Nicholson_solver(
                                func = u_I,
                                mx = 20,
                                mt = 1000,
                                L = 1.0,
                                T = 0.5,
                                kappa = 1.0,
                                u_0 = lambda t_j : np.sin(pi*2*t_j),
                                u_T = lambda t_j : np.sin(pi*2*t_j),
                                bCond = 'Neum'
                                )

####################################################################################

# plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')

L = 1.0; T = 0.5; xx = np.linspace(0,L,250);

pl.title('Crank-Nicholson')
#pl.plot(xx,u_exact(xx, T, kappa = 1.0, L = 1.0),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,{})'.format(T))
pl.legend(loc='upper right')
pl.show()


