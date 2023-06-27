import matplotlib.pyplot as plt
import numpy as np

from roxy.regressor import RoxyRegressor

def my_fun(x, theta):
#    return theta[0] * x ** 2 + theta[1] * x + theta[2]
    return theta[0] * x + theta[1]
    
reg = RoxyRegressor(my_fun)

theta0 = [2, 0.5]
xerr = 0.1
yerr = 0.2
sig = 0.1
    
xtrue = np.linspace(0, 5, 100)
ytrue = reg.value(xtrue, theta0)
xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)
#theta0 = [2, 0.5, -3]

for method in ['uniform', 'profile', 'mnr']:
    print(reg.negloglike(theta0, xobs, yobs, xerr, yerr, sig, method=method))


#y = reg.value(all_x, theta0)
#yp = reg.gradient(all_x, theta0)

#plt.plot(all_x, yp, '.')
#plt.plot(all_x, 2 * theta0[0] * all_x + theta0[1])
plt.plot(xobs, yobs, '.')
plt.plot(xtrue, ytrue)
plt.show()

