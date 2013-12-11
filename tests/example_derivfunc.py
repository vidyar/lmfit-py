#!/usr/bin/env python

from lmfit import Parameters, Minimizer
import numpy as np

try:
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

def func(pars, x, data=None):
	a = pars['a'].value
	b = pars['b'].value
	c = pars['c'].value

	model=a * np.exp(-b * x)+c
	if data is None:
		return model
	return (model - data)

def dfunc(pars, x, data=None):
	a = pars['a'].value
	b = pars['b'].value
	c = pars['c'].value
	v = np.exp(-b*x)
	return [v, -a*x*v, np.ones(len(x))]

def f(var, x):
	return var[0]* np.exp(-var[1] * x)+var[2]

params = Parameters()
params.add('a', value=10)
params.add('b', value=10)
params.add('c', value=10)

a, b, c = 2.5, 1.3, 0.8
x = np.linspace(0,4,50)
y = f([a, b, c], x)
data = y + 0.15*np.random.normal(size=len(x))

# fit without analytic derivative
min1 = Minimizer(func, params, fcn_args=(x,), fcn_kws={'data':data})
min1.leastsq()
fit1 = func(min1.params, x)

# fit with analytic derivative
min2 = Minimizer(func, params, fcn_args=(x,), fcn_kws={'data':data})
min2.leastsq(Dfun=dfunc, col_deriv=1)
fit2 = func(min2.params, x)

print '''Comparison of fit to exponential decay
with and without analytic derivatives, to
   model = a*exp(-b*x) + c
for a = %.2f, b = %.2f, c = %.2f
==============================================
Statistic/Parameter|   Without   | With      |
----------------------------------------------
N Function Calls   |   %3i       |   %3i     |
Chi-square         |   %.4f    |   %.4f  |
   a               |   %.4f    |   %.4f  |
   b               |   %.4f    |   %.4f  |
   c               |   %.4f    |   %.4f  |
----------------------------------------------
''' %  (a, b, c,
        min1.nfev,   min2.nfev,
        min1.chisqr, min2.chisqr,
        min1.params['a'].value, min2.params['a'].value,
        min1.params['b'].value, min2.params['b'].value,
        min1.params['c'].value, min2.params['c'].value )


if HASPYLAB:
	pylab.plot(x, data, 'ro')
	pylab.plot(x, fit1, 'b')
        pylab.plot(x, fit2, 'k')
	pylab.show()



