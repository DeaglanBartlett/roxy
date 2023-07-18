import numpy as np
import matplotlib.pyplot as plt
from roxy.regressor import RoxyRegressor
import roxy.plotting

def my_fun(x, theta):
    return theta[0] * x + theta[1]

param_names = ['A', 'B']
theta0 = [2, 0.5]
param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}


reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

nx = 1000
xerr = 0.01
yerr = 0.01
sig = 0.5
nwarm, nsamp = 700, 5000

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
    
#plt.hist(xtrue, bins=30, density=True, histtype='step')
#plt.show()

ytrue = reg.value(xtrue, theta0)
xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

#plot_kwargs = {'fmt':'.', 'markersize':1, 'zorder':1,
#                 'capsize':1, 'elinewidth':1.0, 'color':'k', 'alpha':1}
#plt.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, **plot_kwargs)
#plt.xlabel(r'$x_{\rm obs}$', fontsize=14)
#plt.ylabel(r'$y_{\rm obs}$', fontsize=14)
#plt.tight_layout()
#plt.show()

reg.optimise(param_names, xobs, yobs, xerr, yerr, method='gmm', ngauss=2)

for method in ['gmm']:
    for i in range(1,3):
        samples = reg.mcmc(param_names, xobs, yobs, xerr, yerr, nwarm, nsamp, method=method, ngauss=2, seed=i)
#    roxy.plotting.trace_plot(samples, to_plot='all', savename=None)
#    roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior, savename=None, show=True)
#    roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr, savename=None)
