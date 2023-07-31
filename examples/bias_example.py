import matplotlib.pyplot as plt
import numpy as np

from roxy.regressor import RoxyRegressor
import roxy.plotting

np.random.seed(4)

# Parameter which vary
Atrue = 15.
sig_true = 0.
Npoints = 10
xerr_mean = 20
exp_scale = 15

# Fixed parameters
Btrue = 1.
yerr_mean = 2
yerr_std = 0.2
xerr_std = xerr_mean / 5

def my_fun(x, theta):
    return theta[0] * x + theta[1]

param_names = ['A', 'B']
theta0 = [Atrue, Btrue]
#param_prior = {'A':[0, 50], 'B':[-50, 50], 'sig':[0, 100]}
param_prior = {'A':[-200, 200], 'B':[-1000, 1000], 'sig':[0, 600]}

reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

xtrue = np.random.exponential(exp_scale, Npoints)
ytrue = reg.value(xtrue, theta0)

xerr = np.random.normal(xerr_mean, xerr_std, Npoints)
xerr[xerr<0.]=0.
    
yerr = np.random.normal(yerr_mean, yerr_std, Npoints)
yerr[yerr<0.]=0.

xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig_true ** 2)

nwarm, nsamp = 700, 10000
samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method='mnr', seed=1234)

roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior, savename='../../Fig5.png')
