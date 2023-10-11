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

import numpy as np

from roxy.regressor import RoxyRegressor

def my_fun(x, theta):
    return theta[0] * x + theta[1]

param_names = ['A', 'B']
theta0 = [2, 0.5]
param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}

reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

nx = 20
xerr = 0.1
yerr = 0.5
sig = 0.5
nwarm, nsamp = 700, 5000

Sxx = np.identity(nx) * xerr ** 2
Sxy = np.zeros((nx,nx))
Syx = np.zeros((nx,nx))
Syy = np.identity(nx) * yerr ** 2

Sigma = np.concatenate(
            [np.concatenate([Sxx, Sxy], axis=-1),
            np.concatenate([Syx, Syy], axis=-1)]
        )
np.random.seed(0)

xtrue = np.linspace(0, 5, nx)
ytrue = reg.value(xtrue, theta0)
xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)


for method in ['mnr', 'unif', 'prof']:

    # Compare the two optimisation routines
    reg.optimise(param_names, xobs, yobs, [xerr, yerr], method=method)
    reg.optimise(param_names, xobs, yobs, Sigma, method=method, covmat=True)
    
    # Compare MCMCs
    reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method)
    reg.mcmc(param_names, xobs, yobs, Sigma, nwarm, nsamp, method=method, covmat=True)
