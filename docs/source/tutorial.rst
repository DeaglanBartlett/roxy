.. default-role:: math

Tutorial
========

In this tutorial we demonstrate how to fit a function to data using both a maximum-likelihood method, and by running a MCMC using the ``roxy`` module. We then plot our results. Throughout we call functions with the argument ``method='mnr'`` as this is the recommended likelihood for data with x and y errors, however this can be replaced with ``method='uniform'`` for an infinite uniform prior on the true x values, or ``method='profile'`` to use the profile likelihood. See the MNR paper for more details on these likelihoods and their advantages/disadvantages.

Defining our function
---------------------

We begin by defining a function which we wish to fit. Here we have ``my_fun``, which is simply a straight line, but more complicated functions can be chosen. The function must take two arguments, the first being the independent variable, and the second are the parameters of interest.

.. code-block:: python

	import matplotlib.pyplot as plt
	import numpy as np
	from roxy.regressor import RoxyRegressor
	import roxy.plotting

	def my_fun(x, theta):
            return theta[0] * x + theta[1]

The optimisation and MCMC functionality of ``roxy`` can be accessed by the ``roxy.regressor.RoxyRegressor`` class, which we define here. 
We must supply the names of each of the parameters of ``my_fun``, as well as a fiducial point and the range of the priors (as a dictionary). We assume uniform priors for all parameters.

.. code-block:: python
	
	param_names = ['A', 'B']
	theta0 = [2, 0.5]
	param_prior = {'A':[0, 5], 'B':[-1, 1], 'sig':[0, 3.0]}

	reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

Mock data generation
--------------------

Let us make some mock data for this function

.. code-block:: python

	nx = 20
	xerr = 0.1
	yerr = 0.5
	sig = 0.5

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

.. image:: data.png
	:width: 480px

Maximum likelihood estimation
-----------------------------

We begin by finding the maximum likelihood point, which is as simple as

.. code-block:: python

	res = reg.optimise(param_names, xobs, yobs, xerr, yerr, method='mnr')

.. code-block:: console

	Optimisation Results:
	A:	2.0954216640049674
	B:	0.18122108584201763
	sig:	0.6317666884191426
	mu_gauss:	2.55679814495946
	w_gauss:	1.4818831988725527

Note that ``res`` here is a ``scipy.optimize._optimize.OptimizeResult`` object, so you can use all the usual functionality this contains.


Markov chain Monte Carlo
------------------------

We will now run a MCMC. This uses the NUTS sampler from ``numpyro`` which is incredibly fast. We choose to use 700 warmup steps and take 5000 samples. We see that the result reports 3613.66 iterations per second, so this MCMC takes less than 2 seconds to run! 

We print the parameter mean and median values, their standard deviations, the 5% and 95% bounds, the number of effective samples and the Gelman-Rubin statistic.

.. code-block:: python

	nwarm, nsamp = 700, 5000
	samples = reg.mcmc(param_names, xobs, yobs, xerr, yerr, nwarm, nsamp, method=method)

.. code-block:: console

	Running MCMC
	sample: 100%|██████████| 5700/5700 [00:01<00:00, 3613.66it/s, 15 steps of size 2.90e-01. acc. prob=0.91]

			mean       std    median      5.0%     95.0%     n_eff     r_hat
		 A      2.09      0.14      2.09      1.85      2.32   3015.95      1.00
		 B      0.19      0.43      0.19     -0.54      0.86   3116.20      1.00
	  mu_gauss      2.55      0.36      2.55      1.98      3.16   3593.55      1.00
	       sig      0.75      0.21      0.73      0.40      1.08   3024.35      1.00
	   w_gauss      1.63      0.28      1.59      1.20      2.09   3000.74      1.00

	Number of divergences: 0

We now plot the results. The trace plot gives the sample value as a function of MCMC step, the triangle plot gives the one- and two-dimensional posterior distributions, and the posterior predictive plot gives the predicted function values at 1, 2 and 3 sigma confidence.
These plots make use of the `arviz <https://www.arviz.org/en/latest/>`_, `getdist <https://getdist.readthedocs.io/en/latest/>`_ and `fgivenx <https://fgivenx.readthedocs.io/en/latest/?badge=latest>`_ modules, respectively. We also have functionality to produce triangle plots with the `corner <https://corner.readthedocs.io/en/latest/>`_ module (by replacing ``module='getdist'`` with ``module='corner'`` in ``roxy.plotting.triangle_plot``).

.. code-block:: python

	roxy.plotting.trace_plot(samples, to_plot='all')
	roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior)
	roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr) 

.. image:: trace.png
        :width: 480px

.. image:: triangle.png
        :width: 480px

.. image:: posterior_predictive.png
        :width: 480px
