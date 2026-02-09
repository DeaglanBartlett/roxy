import numpy as np
from roxy.regressor import RoxyRegressor
import roxy.plotting as plotting

def test_example1():

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

    reg.optimise(param_names, xobs, yobs, [xerr, yerr], method='unif')

    nwarm, nsamp = 700, 5000
    reg.mcmc(param_names, xobs, yobs, [xerr, yerr],
            nwarm, nsamp, method='mnr')

    return
    
    
def test_example2():

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

    nwarm, nsamp = 700, 5000
    reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp,
            method='gmm', ngauss=2, gmm_prior='uniform')

    max_ngauss = 3
    np.random.seed(42)
    reg.find_best_gmm(param_names, xobs, yobs, xerr, yerr, max_ngauss,
            best_metric='BIC', nwarm=100, nsamp=100, gmm_prior='uniform')
            
    return



def test_example3():
    """
    Example for the upper limit functionality. 
    """

    
    np.random.seed(0)


    def my_fun(x, theta):
        return theta[0] * x + theta[1]

    param_names = ['A', 'B']
    theta0 = [2, 0.5]
    param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}

    reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)
    

    nx = 50
    xerr = 0.1
    yerr = 0.5
    sig = 0.8

    xtrue = np.linspace(0, 5, nx)
    ytrue = reg.value(xtrue, theta0)
    
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)


    det_threshold = 4.0
    # mask = True where value is an *upper limit* / censored
    mask = yobs <= det_threshold
    # Replace censored yobs with the threshold (no in-place assignment)
    yobs = np.where(mask, det_threshold, yobs)
    # Detection flag: True if detected, False if upper-limit
    y_is_detected = ~mask

    nwarm, nsamp = 700, 5000
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp,
            method='mnr', y_is_detected=y_is_detected)
    
    
    plotting.trace_plot(samples, to_plot='all')
    plotting.triangle_plot(samples, to_plot='all', module='getdist', param_prior=param_prior)
    #plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr, y_is_detected=y_is_detected)