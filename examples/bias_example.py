import matplotlib.pyplot as plt
import numpy as np

from roxy.regressor import RoxyRegressor
import roxy.plotting
import roxy.mcmc

np.random.seed(4)

# Parameter which vary
# [Atrue, sig_true, Npoints, xerr_mean, exp_scale]
all_param = [
    [15.0, 0.0, 10, 20.0, 15.0],  # Figure 5
    [-15.0, 10.0, 894, 20.0, 15.0],
    [15.0, 0.0, 4000, 15.0, 15.0],
    [15.0, 0.0, 4000, 20.0, 15.0],
    [-15.0, 5.0, 4000, 20.0, 15.0],
    [15.0, 0.0, 10, 20.0, 15.0],
    [15.0, 10.0, 4000, 5.0, 15.0],
    [2, 1000, 5, 2.3, 8],
    [2, 1000, 5, 20, 8],
]

# Fixed parameters
Btrue = 1.
yerr_mean = 2
yerr_std = 0.2

# MCMC params
#nwarm, nsamp = 700, 500
nwarm, nsamp = 700, 10000

def my_fun(x, theta):
    return theta[0] * x + theta[1]
param_names = ['A', 'B']

#param_prior = {'A':[0, 50], 'B':[-50, 50], 'sig':[0, 100]}
param_prior = {'A':[-200, 200], 'B':[-1000, 1000], 'sig':[0, 600]}
#param_prior = {'A':[None, None], 'B':[None, None], 'sig':[None, None]}

for par in all_param[:1]:
    Atrue, sig_true, Npoints, xerr_mean, exp_scale = par
    theta0 = [Atrue, Btrue]
    xerr_std = xerr_mean / 5

    reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

    # Make data
    xtrue = np.random.exponential(exp_scale, Npoints)
    ytrue = reg.value(xtrue, theta0)
    xerr = np.random.normal(xerr_mean, xerr_std, Npoints)
    xerr[xerr<0.]=0.
    yerr = np.random.normal(yerr_mean, yerr_std, Npoints)
    yerr[yerr<0.]=0.
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig_true ** 2)

    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method='mnr', seed=1234)
    
    truths = {'A':Atrue, 'B':Btrue, 'sig':sig_true}
    biases = roxy.mcmc.compute_bias(samples, truths)
    
    roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior, savename=None)
#    roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior, savename='../../Fig5.png')
