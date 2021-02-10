import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import math
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


def tpoints(t0 = 0, t1 = 100, h = 1e-3): 
	# time points for plotting

	return np.linspace(t0,t1,int(t1/h))


def Hopf_bi_NF(U, t, beta):
	"""
	Specific ODE to be analysed.

	USAGE:
	  -   input the dimensions of the ODE into the 'if' statement
	  -   unpack U with the same dimensions e.g. u1, u2, u3 = U for 3
	  -   edit u_dot equations for desired functions
	  -   output through vector V

	  -   make sure inputs match equation parameters

	"""

	# edit out U, and u_dot equations for different ODEs

	if len(U) == 2: # check dimensions match the ODE, if 3 'u' equations,
					# 'if len(U) == 3:' and so on...

		sigma = -1

		u1, u2 = U

		u1_dot = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
		u2_dot = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

		V = np.array([u1_dot, u2_dot])
	else:
		raise Exception('\n 	Dimensions of initial values don’t match those of the ODE')

	return V


def vals(X0, ODE_func, tspan, par):
	# x, y values formatted for plotting and functions

	sols = odeint(ODE_func, X0, tspan, args = (par,))

	return sols


def index_minima(x):
	# provides the indexs where the minima lie

	dips = np.where((x[1:-1] < x[:-2]) * (x[1:-1] < x[2:]))[0] + 1

	return dips


def period_calc(x, tspan):
	# period T is calculated using differences
	#				 of times where minima occurs

	dips = index_minima(x)

	if dips.size > 1: # 2 minima needed to calculate time period
		tix = tspan[dips]
		periodT = tix[-1] - tix[-2]
	else:
		raise TypeError('\n 	Attempted ODE is non-periodic')

	return round(periodT,3)


def fun(X0, ODE_func, periodT, par):
	"""
	USER INPUT:
		Here u2_dot is the phase condition that is to be set to zero.

		If the phase condition has to be changed edit it here.
		   - [0] for u1, [1] for u2, etc.

	"""

	# functions for root finding 

	tspan_period = tpoints(t1 = periodT)
	# U(0) = U(T)
	osc = odeint(ODE_func, X0, tspan_period, args = (par,))[-1,1] - X0[1]
	# phase condition:
	u2_dot = ODE_func(X0, tspan_period, par)[1] # du2dt(0) = 0

	return [osc, u2_dot]


def display(X0, ODE_func, par):
	"""
    A function that uses python integrators and root finders to 
    achieve an accurate solution to an input ODE.

    Parameters
    ----------
    X0 : numpy.array
        An initial guess at the initial values of the ODE solution.

    ODE_func : function
        The ODE to be solved. The ODE function takes
        a single parameter and return the right-hand side of
        the ODE as a numpy.array.

    par : parameter
    	The parameter that the ODE takes

    Returns
    -------
    Returns a numpy.ndarray of the solutions of the input ODE, 
    and the period of the oscillatory function.
    """

	tspan = tpoints()
	sols = vals(X0, ODE_func, tspan, par)
	periodT = period_calc(sols[:,1], tspan)
	X0_solved = fsolve(fun, X0, args = (ODE_func, periodT, par))
	# solutions for improved initial conditions
	sols = vals(X0_solved, ODE_func, tspan, par)
	
	return sols, periodT


def param_cont(P, X0, ODE_func):
	"""
    A function that uses numerical shooting to find limit cycles of
    a specified ODE.

    Parameters
    ----------
	P : Parameter range
		A linspace of the range of paramater values that 
		limit cycles should be found over

    X0 : numpy.array
        An initial guess at the initial values for the limit cycle.

    ODE_func : function
        The ODE to apply shooting to. The ode function takes
        a single parameter and return the right-hand side of
        the ODE as a numpy.array.

    Returns
    -------
    Returns a numpy.ndarray of the solutions of the input ODE, 
    and the period of the oscillatory function.
    """

	X0_final = X0
	X0_plot = np.zeros((len(P),len(X0))) 

	for i in range(len(P)):
		p = (P[i])
		try:
			s,_ = display(X0_final ,ODE_func, p) # takes only the solutions
		except TypeError: # non periodic
			return P, X0_plot
		
		X0_final = s[0,:]
		X0_plot[i,:] = [np.amin(s[:,1]),np.amax(s[:,1])]

	return P, X0_plot

##	comment function calls below out for faster running, for example when testing
# sols, periodT = display(np.array([0.35, 0.35]),Hopf_bi_NF, par = 1)
# P, X0_plot = param_cont(np.linspace(2,-1,1000), np.array([0.35, 0.35]), Hopf_bi_NF)


# # PLOTTING # # uncomment all to produce both graphs 

# plt.plot(P,X0_plot)
# plt.show()

# fig, ax1 = plt.subplots(1, 1)

# for i in range(sols.shape[1]):
# 	ax1.plot(tpoints(), sols[:,i], label = 'u{}'.format(i+1))

# ax1.set_title('ODE')
# ax1.set_xlabel(r'$Time$')
# ax1.set_ylabel(r'$U(t)$')
# ax1.set_xlim([int(min(tpoints())), int(max(tpoints()))])
# ax1.grid()
# ax1.legend(loc = 'upper right')

# plt.show()

