from usage import u_I, u_exact, Crank_Nicholson_solver

import random
import pylab as pl
import numpy as np

#	random variables to show strength/durability of code
mx = random.randint(5,20)
mt = random.randint(50,1000)


x, u_j = Crank_Nicholson_solver(
                                func = u_I,
                                mx = mx,
                                mt = mt,
                                L = 1.0,
                                T = 0.5,
                                kappa = 1.0,
                                u_0 = 0,
                                u_T = 0,
                                bCond = 'D_hom'
                                )

indx = random.randint(1,mx); x_indx = x[indx]; u_j_indx = u_j[indx]
sol = u_exact(x_indx, t = 0.5, kappa = 1.0, L = 1.0)

#	show the user the variables that were randomised
print('\n', 'mx =',mx, 'mt =',mt, '\n')

print('Test for error:')
# condition
err = abs(sol - u_j_indx)
print('Error =', err)
if err < 1e-2: # tolerance
	print("	Test successful\n")
else:
	print("	Test failed\n")

################################################################################

# plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')

L = 1.0; T = 0.5; xx = np.linspace(0,L,250);

pl.title('Crank-Nicholson')
pl.plot(xx,u_exact(xx, T, kappa = 1.0, L = 1.0),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,{})'.format(T))
pl.legend(loc='upper right')
pl.show