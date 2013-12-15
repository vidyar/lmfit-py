from lmfit import Parameters, minimize, report_fit

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign

try:
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('period', value=5.33)
p_true.add('shift', value=0.123)
p_true.add('decay', value=0.010)

def residual(pars, x, data=None):
    amp = pars['amp'].value
    per = pars['period'].value
    shift = pars['shift'].value
    decay = pars['decay'].value

    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = amp*sin(shift + x/per) * exp(-x*x*decay*decay)
    if data is None:
        return model
    return (model - data)

n = 2500
xmin = 0.
xmax = 250.0
noise = random.normal(scale=0.7215, size=n)
x     = linspace(xmin, xmax, n)
data  = residual(p_true, x) + noise

pars = Parameters()
pars.add('amp', value=9.0, min=5, max=20)
pars.add('period', value=3., min=1., max=7)
pars.add('shift', value=-.10,  min=-0.2, max=0.2)
pars.add('decay', value=2.e-3, min=0, max=0.1)

init = residual(pars, x)

out = minimize(residual, pars, method='lbfgsb', args=(x,), kws={'data':data})

fit = residual(out.params, x)

for name, par in out.params.items():
    nout = "%s:%s" % (name, ' '*(20-len(name)))
    print "%s: %s (%s) " % (nout, par.value, p_true[name].value)

print out.chisqr,  out.nfree
#
report_fit(out.params, modelpars=p_true)

if HASPYLAB:
    pylab.plot(x, data, 'r--')
    pylab.plot(x, init, 'k')
    pylab.plot(x, fit, 'b')
    pylab.show()




