.. default-role:: math

Tutorial
========


.. code-block:: python

	import matplotlib.pyplot as plt
	import numpy as np
	from roxy.regressor import RoxyRegressor
	import roxy.plotting

	def my_fun(x, theta):
        	return theta[0] * x + theta[1]

	param_names = ['A', 'B']
	theta0 = [2, 0.5]
	param_prior = {'A':[0, 5], 'B':[-1, 1], 'sig':[0, 3.0]}

	reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

Make some mock data

.. code-block:: python

	nx = 20
	xerr = 0.1
	yerr = 0.5
	sig = 0.5

	xtrue = np.linspace(0, 5, nx)
	ytrue = reg.value(xtrue, theta0)
	xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
	yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

Find the maximum likelihood point

.. code-block:: python

	reg.negloglike(theta0, xobs, yobs, xerr, yerr, sig, method='mnr')

Run an MCMC and plot the results

.. code-block:: python

	samples = reg.mcmc(param_names, xobs, yobs, xerr, yerr, nwarm, nsamp, method=method)
	roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior)
	roxy.plotting.trace_plot(samples, to_plot='all')
	roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr) 
