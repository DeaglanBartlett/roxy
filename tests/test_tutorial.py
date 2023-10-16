import numpy as np
from roxy.regressor import RoxyRegressor
import roxy.plotting
import roxy.mcmc
import roxy.likelihoods
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

def test_example1(monkeypatch):

    monkeypatch.setattr(plt, 'show', lambda: None)

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
    plt.clf()
    plt.close(plt.gcf())

    reg.optimise(param_names, xobs, yobs, [xerr, yerr], method='unif')

    nwarm, nsamp = 700, 5000
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr],
            nwarm, nsamp, method='mnr')

    roxy.plotting.trace_plot(samples, to_plot='all', savename=None)
    roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist',
                param_prior=param_prior, savename=None)
    roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs,
                xerr, yerr, savename=None)
                
    return
    
    
def test_example2(monkeypatch):

    monkeypatch.setattr(plt, 'show', lambda: None)

    np.random.seed(0)

    nx = 1000

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
    fig.clf()

    def my_fun(x, theta):
        return theta[0] * x + theta[1]

    param_names = ['A', 'B']
    theta0 = [2, 0.5]
    param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}
    xerr = 0.1
    yerr = 0.5
    sig = 0.5

    reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

    ytrue = reg.value(xtrue, theta0)
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

    plot_kwargs = {'fmt':'.', 'markersize':1, 'zorder':1,
             'capsize':1, 'elinewidth':1.0, 'color':'k', 'alpha':1}
    plt.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, **plot_kwargs)
    plt.xlabel(r'$x_{\rm obs}$', fontsize=14)
    plt.ylabel(r'$y_{\rm obs}$', fontsize=14)
    plt.clf()

    nwarm, nsamp = 700, 5000
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp,
            method='gmm', ngauss=2, gmm_prior='uniform')
    roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist',
            param_prior=param_prior, show=False, savename='gmm_corner.png')

    max_ngauss = 3
    np.random.seed(42)
    reg.find_best_gmm(param_names, xobs, yobs, xerr, yerr, max_ngauss,
            best_metric='BIC', nwarm=100, nsamp=100, gmm_prior='uniform')
            
    return
