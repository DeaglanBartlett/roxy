# Copyright 2023 Deaglan J. Bartlett
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in the
# Software without restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

import numpy as np
import matplotlib.pyplot as plt
from roxy.regressor import RoxyRegressor

def my_fun(x, theta):
    return theta[0] * x + theta[1]

param_names = ['A', 'B']
theta0 = [2, 0.5]
param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}


reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

nx = 1000
xerr = 0.1
yerr = 0.5
sig = 0.5
nwarm, nsamp = 700, 10000

np.random.seed(0)

# Draw the samples from a two Gaussian model
true_weights = np.array([0.7, 0.3])
true_means = [-10.0, 0.0]
true_w = [2, 3]
which_gauss = np.random.uniform(0, 1, nx)

p = np.array([0] + list(true_weights))
p = np.cumsum(p)
xtrue = np.empty(nx)
for i in range(len(true_means)):
    m = (which_gauss >= p[i]) & (which_gauss < p[i+1])
    print(i, m.sum())
    xtrue[m] = np.random.normal(true_means[i], true_w[i], m.sum())
    
fig, ax = plt.subplots(1, 1, figsize=(10,4))
ax.hist(xtrue, bins=30, density=True, histtype='step', color='b')
x = np.linspace(xtrue.min(), xtrue.max(), 300)
ysum = np.zeros(len(x))
for nu, mu, w in zip(true_weights, true_means, true_w):
    y = nu / np.sqrt(2 * np.pi * w ** 2) * np.exp(- (x - mu) ** 2 / (2 * w ** 2))
    ysum += y
    ax.plot(x, y, color='k')
ax.plot(x, ysum, color='r', ls='--')
ax.set_xlabel(r'$x_{\rm t}$')
ax.set_ylabel(r'$p(x_{\rm t})$')
fig.tight_layout()
fig.savefig('../docs/source/gmm_distribution.png', transparent=True)
fig.clf()
plt.close(fig)

ytrue = reg.value(xtrue, theta0)
xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

plot_kwargs = {'fmt':'.', 'markersize':1, 'zorder':1,
                 'capsize':1, 'elinewidth':1.0, 'color':'k', 'alpha':1}
plt.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, **plot_kwargs)
plt.xlabel(r'$x_{\rm obs}$', fontsize=14)
plt.ylabel(r'$y_{\rm obs}$', fontsize=14)
plt.tight_layout()
plt.savefig('../docs/source/gmm_data.png', transparent=True)
plt.clf()
plt.close(plt.gcf())

max_ngauss = 3
gmm_prior = 'hierarchical'
reg.find_best_gmm(param_names, xobs, yobs, xerr, yerr, max_ngauss, best_metric='BIC',
    nwarm=100, nsamp=100, gmm_prior=gmm_prior)

all_method = ['prof', 'unif', 'mnr']
all_bic = [None] * len(all_method)
all_nll = [None] * len(all_method)
for i, method in enumerate(all_method):
    all_nll[i], all_bic[i] = reg.compute_information_criterion('BIC', param_names,
        xobs, yobs, [xerr, yerr], nwarm=100, nsamp=100, method=method)
for method, nll, bic in zip(all_method, all_nll, all_bic):
    print(method, nll, bic)

