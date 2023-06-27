import matplotlib.pyplot as plt
import numpy as np

from roxy.regressor import RoxyRegressor
import roxy.plotting

run_name = 'linear'
#run_name = 'quadratic'

if run_name == 'linear':

    def my_fun(x, theta):
        return theta[0] * x + theta[1]

    param_names = ['A', 'B']
    theta0 = [2, 0.5]
    param_prior = {'A':[0, 5], 'B':[-1, 1], 'sig':[0, 3.0]}
    
elif run_name == 'quadratic':

    def my_fun(x, theta):
        return theta[0] * x ** 2 + theta[1] * x + theta[2]
    
    param_names = ['A', 'B', 'C']
    theta0 = [2, 0.5, -3]
    param_prior = {'A':[0, 5], 'B':[-1, 1], 'C': [-10, 10], 'sig':[0, 3.0]}

reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

nx = 20
xerr = 0.1
yerr = 0.5
sig = 0.5
nwarm, nsamp = 700, 5000
    
xtrue = np.linspace(0, 5, nx)
ytrue = reg.value(xtrue, theta0)
xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)
#theta0 = [2, 0.5, -3]

#for method in ['uniform', 'profile', 'mnr']:
for method in ['mnr']:
    print(reg.negloglike(theta0, xobs, yobs, xerr, yerr, sig, method=method))
#    reg.optimise(['A'], xobs, yobs, xerr, yerr, method=method)
    samples = reg.mcmc(param_names, xobs, yobs, xerr, yerr, nwarm, nsamp, method=method)
    roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior)
    roxy.plotting.trace_plot(samples, to_plot='all')
    roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr)


#y = reg.value(all_x, theta0)
#yp = reg.gradient(all_x, theta0)

#plt.plot(all_x, yp, '.')
#plt.plot(all_x, 2 * theta0[0] * all_x + theta0[1])
#plt.plot(xobs, yobs, '.')
#plt.plot(xtrue, ytrue)
#plt.show()

