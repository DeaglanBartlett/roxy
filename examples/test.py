import matplotlib.pyplot as plt
import numpy as np

from roxy.regressor import RoxyRegressor

def my_fun(x, theta):
#    return theta[0] * x ** 2 + theta[1] * x + theta[2]
    return theta[0] * x + theta[1]

param_names = ['A', 'B']
theta0 = [2, 0.5]
param_prior = {'A':[0, 5], 'B':[-1, 1], 'sig':[0, 0.3]}

reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

xerr = 0.1
yerr = 0.2
sig = 0.1
    
xtrue = np.linspace(0, 5, 100)
ytrue = reg.value(xtrue, theta0)
xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)
#theta0 = [2, 0.5, -3]

#for method in ['uniform', 'profile', 'mnr']:
for method in ['mnr']:
    print(reg.negloglike(theta0, xobs, yobs, xerr, yerr, sig, method=method))
    reg.optimise(['A'], xobs, yobs, xerr, yerr, method=method)


#y = reg.value(all_x, theta0)
#yp = reg.gradient(all_x, theta0)

#plt.plot(all_x, yp, '.')
#plt.plot(all_x, 2 * theta0[0] * all_x + theta0[1])
plt.plot(xobs, yobs, '.')
plt.plot(xtrue, ytrue)
plt.show()

