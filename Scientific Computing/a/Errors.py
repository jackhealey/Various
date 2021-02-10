import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sys

from myForwardEuler1DDiffusion import Forward_Euler_solver
from myBackwardEuler import u_I, u_exact, Backward_Euler_solver
from usage import Crank_Nicholson_solver

def Simpson_Third(Solver,		func = u_I,
                                mx = 1000,
                                mt = 1000,
                                L = 1.0,
                                T = 0.5,
                                kappa = 1.0,
                                u_0 = 0,
                                u_T = 0,
                                bCond = 'D_hom'
                                ):

	x, u_j = Solver(func, mx, mt, L, T, kappa, u_0, u_T, bCond)

	indx = np.linspace(1, mx-1, mx-1)
	odds = indx[0::2]
	evens = indx[1::2]

	odds_sum = 0
	evens_sum = 0
	for i in odds:
		odds_sum += u_j[int(i)]

	for j in evens:
		evens_sum += u_j[int(j)]

	if bCond == 'D_hom':
		apprx_Int = (L/(3*mx)) * (u_0 + 4*(odds_sum) + 2*(evens_sum) + u_T)
	else:
		raise Exception('\nErrors work for homogenous Dirichlet conditions only')

	return apprx_Int

def Error_plot(n, T, kappa, L):
	# n: number of points in logspace

	h = np.zeros(n)
	E_simp = np.zeros((n,3))
	i = 0

	exact_Int,_ = quad(u_exact, 0, L, args = (T, kappa, L))

	for mt in (np.logspace(1,n,n)):
		deltat = T/mt

		E_FE = abs(Simpson_Third(Solver = Forward_Euler_solver,
												 mt = int(mt)) - exact_Int)
		E_BE = abs(Simpson_Third(Solver = Backward_Euler_solver,
												 mt = int(mt)) - exact_Int)
		E_CN = abs(Simpson_Third(Solver = Crank_Nicholson_solver,
												 mt = int(mt)) - exact_Int)

		if 'p_errors' in sys.argv[:]:
			print()
			print('Simpson\'s rule integration: mt =' ,int(mt), ', deltat =', deltat)
			print('Forward Euler Method:  ', E_FE)
			print('Backward Euler Method: ', E_BE)
			print('Crank Nicholson Method:', E_CN)

		E_simp[i,:] = np.array([E_FE, E_BE, E_CN])
		h[i] = deltat
		i+=1

	return E_simp, h

def Error_order(E_simp, h):

	E_ord = np.zeros(len(h))
	for i in range(len(h)):
		E_ord[i] = (math.log(E_simp[0,i]/E_simp[1,i])) / (math.log(h[0]/h[1]))
	
	return E_ord

E_simp, h = Error_plot(n = 3, T = 0.5, kappa = 1.0, L = 1.0)

E_ord = Error_order(E_simp, h)

labels = ['Forward Euler', 'Backward Euler', 'Crank Nicholson']
for i in range(len(h)):
	print('Error order', labels[i], '=', E_ord[i])

fig1, ax1 = plt.subplots()

for i in range(1, E_simp.shape[1]):
	ax1.loglog(h, E_simp[:,i], 'o-', label = labels[i])

ax1.set_title(r"A log-log graph of E against $\Delta t$ using Simpson's third rule to approximate the integral.")
ax1.set_xlabel(r'$\Delta t$')
ax1.set_ylabel(r'Absolute Error')
ax1.legend(loc='upper right')
plt.show()



