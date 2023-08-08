# Copyright 2023 Deaglan J. Bartlett
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.

import matplotlib.pyplot as plt
import numpy as np

from roxy.regressor import RoxyRegressor
import roxy.plotting

#run_name = 'linear'
run_name = 'quadratic'

if run_name == 'linear':

    def my_fun(x, theta):
        return theta[0] * x + theta[1]

    param_names = ['A', 'B']
    theta0 = [2, 0.5]
    param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}
    
elif run_name == 'quadratic':

    def my_fun(x, theta):
        return theta[0] * x ** 2 + theta[1] * x + theta[2]
    
    param_names = ['A', 'B', 'C']
    theta0 = [2, 0.5, -3]
    param_prior = {'A':[None, None], 'B':[None, None], 'C': [None, None], 'sig':[None, None]}

reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

nx = 20
xerr = 0.1
yerr = 0.5
sig = 0.5
nwarm, nsamp = 700, 5000

np.random.seed(0)
    
xtrue = np.linspace(0, 5, nx)
ytrue = reg.value(xtrue, theta0)
xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

plot_kwargs = {'fmt':'.', 'markersize':1, 'zorder':1,
                 'capsize':1, 'elinewidth':1.0, 'color':'k', 'alpha':1}
plt.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, **plot_kwargs)
plt.xlabel(r'$x_{\rm obs}$', fontsize=14)
plt.ylabel(r'$y_{\rm obs}$', fontsize=14)
plt.tight_layout()
#plt.savefig('../docs/source/data.png', transparent=True)
plt.show()
plt.clf()
plt.close(plt.gcf())

reg.optimise(param_names, xobs, yobs, [xerr, yerr], method='mnr')

#theta0 = [2, 0.5, -3]

#for method in ['unif', 'prof', 'mnr']:
for method in ['mnr']:
    print(reg.negloglike(theta0, xobs, yobs, [xerr, yerr], sig, method=method))
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method, num_chains=2)
    roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior,) #savename='../docs/source/triangle.png')
    roxy.plotting.trace_plot(samples, to_plot='all',) #savename='../docs/source/trace.png')
    roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr) #, savename='../docs/source/posterior_predictive.png')


#y = reg.value(all_x, theta0)
#yp = reg.gradient(all_x, theta0)

#plt.plot(all_x, yp, '.')
#plt.plot(all_x, 2 * theta0[0] * all_x + theta0[1])
#plt.plot(xobs, yobs, '.')
#plt.plot(xtrue, ytrue)
#plt.show()

