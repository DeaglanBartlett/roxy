import numpy as np
from roxy.regressor import RoxyRegressor
import roxy.plotting
import roxy.mcmc
import roxy.causality
import jax.random
import jax.numpy as jnp
import roxy.likelihoods
import matplotlib.pyplot as plt
import unittest



def test_example_standard(monkeypatch):

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
    
    truths = {p:v for p,v in zip(param_names, theta0)}
    truths['sig'] = sig

    np.random.seed(0)
    xtrue = np.linspace(0.01, 5, nx)
    ytrue = reg.value(xtrue, theta0)
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

    nwarm, nsamp = 70, 500
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr],
                nwarm, nsamp, method='mnr')

    # Default plotting
    roxy.plotting.trace_plot(samples, to_plot='all', savename='trace.png',
        show=True)
    roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist',
            param_prior=param_prior, savename='corner.png', show=True)
    
    xlim = (xobs.min()*0.8, xobs.max()*1.2)
    ylim = (yobs.min()*0.8, yobs.max()*1.2)
    for xscale in ['linear', 'log']:
        roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr,
            show=True, savename='predictive.png',
            xlim=xlim, ylim=ylim, xscale=xscale, yscale=xscale)
        
    # Just plot some variables
    roxy.plotting.triangle_plot(samples, to_plot=['A', 'B'], module='getdist',
        param_prior=param_prior, savename=None, show=False)
    roxy.plotting.trace_plot(samples, to_plot=['A', 'B'], savename=None, show=False)
    
    # Param prior checks
    roxy.plotting.triangle_plot(samples, to_plot=['A', 'B'], module='getdist',
        param_prior=None, savename=None, show=False)
    param_prior['A'] = [None, None]
    param_prior['sig'] = [0, None]
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr],
                nwarm, nsamp, method='mnr')
    roxy.plotting.triangle_plot(samples, to_plot=['A', 'B'], module='getdist',
        param_prior=param_prior, savename=None, show=False)
        
    # Check warnings when prior too narrow
    param_prior['A'] = [0.0, 1.0]
    param_prior['sig'] = [0.0, 3.0]
    reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method='mnr')
    param_prior['A'] = [0.0, 5.0]
            
    # Check labels
    roxy.plotting.triangle_plot(samples, to_plot=['A', 'B'], module='getdist',
        param_prior=param_prior, savename=None, show=False, labels={'A':'A', 'B':'B'})
      
    # Check corner also works
    roxy.plotting.triangle_plot(samples, to_plot='all', module='corner',
        param_prior=param_prior, savename=None, show=False)
        
    try:
        roxy.plotting.triangle_plot(samples, to_plot='all', module='badmodule',
            param_prior=param_prior, savename=None, show=False)
    except NotImplementedError:
        pass
        
    # Test biases
    roxy.mcmc.compute_bias(samples, truths, verbose=True)
    
    # Check MCMC without intrinsic scatter
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr],
                nwarm, nsamp, method='mnr', infer_intrinsic=False)
    
    # A few likelihood checks
    for Ai in [theta0[0], [theta0[0]]]:
        roxy.likelihoods.negloglike_mnr(xobs, yobs, xerr, yerr, ytrue,
            Ai, sig, 2.5, 1.0)
        roxy.likelihoods.negloglike_gmm(xobs, yobs, xerr, yerr, ytrue,
            Ai, sig, [2.5], [1.0], [1.0])
        roxy.likelihoods.negloglike_unif(xobs, yobs, xerr, yerr, ytrue,
            Ai, sig)
            
    # Test regressor negloglike
    reg.negloglike(theta0, xobs, yobs, [xerr, yerr], sig=sig,
                mu_gauss=2.5, w_gauss=1.0, test_prior=False)
    try:
        reg.negloglike(theta0, xobs, yobs, [xerr, yerr], sig=sig,
                mu_gauss=2.5, w_gauss=1.0, method='gmm', covmat=True)
    except NotImplementedError:
        pass
    try:
        reg.negloglike(theta0, xobs, yobs, [xerr, yerr], sig=sig,
                mu_gauss=2.5, w_gauss=1.0, method='unknown')
    except NotImplementedError:
        pass
    assert np.isnan(reg.negloglike(theta0, xobs, yobs, [xerr, yerr], sig=-1)), \
            "Negative sigma should give nan loglike"
        
    # Test regressor optimise
    reg.get_param_index(['A'])
    reg.optimise(param_names, xobs, yobs, [xerr, yerr], method='unif',
            infer_intrinsic=True)
    reg.optimise(param_names, xobs, yobs, [xerr, yerr], method='unif',
            infer_intrinsic=False)
        
    return
    
    
def test_example_gmm():
    
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

    nwarm, nsamp = 70, 500
    for gmm_prior in ['uniform', 'hierarchical']:
        samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp,
                        method='gmm', ngauss=2, gmm_prior=gmm_prior)
        roxy.plotting.triangle_plot(samples, to_plot='all', module='getdist',
                        param_prior=param_prior, show=False, savename=None)
        roxy.plotting.trace_plot(samples, to_plot='all', savename=None, show=False)
        roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr,
            show=False, savename=None)
     
    # Check unknown priors raise exceptions
    try:
        reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp,
                        method='gmm', ngauss=2, gmm_prior='unknown')
    except NotImplementedError:
        pass

    max_ngauss = 3
    np.random.seed(42)
    ngauss = reg.find_best_gmm(param_names, xobs, yobs, xerr, yerr, max_ngauss,
                best_metric='BIC', nwarm=100, nsamp=100, gmm_prior='uniform')
    assert ngauss == 2, "Did not find 2 Gaussians for case which clearly needs 2"
         
    # Check different criteria work as expected
    for criterion in ['AIC', 'BIC']:
        reg.compute_information_criterion(criterion, param_names, xobs, yobs,
            [xerr, yerr], ngauss=1)
    try:
        reg.compute_information_criterion('DIC', param_names, xobs, yobs,
            [xerr, yerr], ngauss=1)
    except NotImplementedError:
        pass
        
    # Check information criterion for hierarchical prior
    reg.compute_information_criterion('BIC', param_names, xobs, yobs,
            [xerr, yerr], ngauss=1, method='gmm', gmm_prior='hierarchical')
            
    # Check GMM with covmat raises exception
    Sxx = np.identity(nx) * xerr ** 2
    Sxy = np.zeros((nx,nx))
    Syx = np.zeros((nx,nx))
    Syy = np.identity(nx) * yerr ** 2
    Sigma = np.concatenate(
                [np.concatenate([Sxx, Sxy], axis=-1),
                np.concatenate([Syx, Syy], axis=-1)]
            )
    try:
        reg.mcmc(param_names, xobs, yobs, Sigma, nwarm, nsamp,
                        method='gmm', ngauss=2, covmat=True)
    except NotImplementedError:
        pass
                
    return
    
    
def test_example_exp():

    def my_fun(x, theta):
        return jnp.exp(theta[0] * x)
    
    # Choose parameters so second derivative large to raise warning
    param_names = ['A']
    theta0 = [1.5]
    param_prior = {'A':[1.0, 3.0], 'sig':[0, 1]}
                
    reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

    nx = 100
    xerr = 1.5
    yerr = 0.5
    sig = 0.5
    nwarm, nsamp = 70, 500

    np.random.seed(0)
        
    xtrue = np.linspace(0, 1, nx)
    ytrue = reg.value(xtrue, theta0)
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

    nwarm, nsamp = 70, 500
    reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method='mnr')
                
    return
    
    
def test_different_likes():

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
    nwarm, nsamp = 70, 500

    np.random.seed(0)
        
    xtrue = np.linspace(0, 5, nx)
    ytrue = reg.value(xtrue, theta0)
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)
    
    Sxx = np.identity(nx) * xerr ** 2
    Sxy = np.zeros((nx,nx))
    Syx = np.zeros((nx,nx))
    Syy = np.identity(nx) * yerr ** 2
    Sigma = np.concatenate(
                [np.concatenate([Sxx, Sxy], axis=-1),
                np.concatenate([Syx, Syy], axis=-1)]
            )
    
    for method in ['unif', 'prof', 'mnr', 'gmm']:
        print(method)
        if method == 'gmm':
            for p in ['uniform', 'hierarchical']:
                print(p)
                reg.optimise(param_names, xobs, yobs, [xerr, yerr],
                    method=method, gmm_prior=p)
            p = 'unknown'
            try:
                reg.optimise(param_names, xobs, yobs, [xerr, yerr],
                    method=method, gmm_prior=p)
            except NotImplementedError:
                pass
        else:
            reg.optimise(param_names, xobs, yobs, [xerr, yerr], method=method)
            reg.optimise(param_names, xobs, yobs, Sigma, method=method, covmat=True)
            reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method)
            reg.mcmc(param_names, xobs, yobs, Sigma, nwarm, nsamp, method=method,
                    covmat=True)
    
    # Test unknown method raises exception in MCMC
    try:
        reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method='unknown')
    except NotImplementedError:
        pass

    return
    


def test_example_with_uplims():
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
    xtrue = np.linspace(0.01, 5, nx)
    ytrue = reg.value(xtrue, theta0)
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)
    
    # Make some upper limits
    y_is_detected = np.ones_like(yobs).astype(bool)
    y_is_detected[::5] = False


    # Check that passing an incorrect y_is_detected raises an error
    try:
        reg.optimise(param_names, xobs, yobs, [xerr, yerr], method='mnr', y_is_detected=[0,1])
    except ValueError:
        pass


    reg.optimise(param_names, xobs, yobs, [xerr, yerr], method='mnr', y_is_detected=y_is_detected)

    nwarm, nsamp = 70, 500
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr],
                nwarm, nsamp, method='mnr', y_is_detected=y_is_detected)
    assert isinstance(samples, dict), "MCMC should return a dictionary of samples"
    for k in param_names + ['sig', 'mu_gauss', 'w_gauss']:
        assert k in samples, f"Samples should contain key {k}"
        assert len(samples[k]) == nsamp, f"Samples for {k} should have length {nsamp}"
    
    return
    

    
def test_mcmc_classes():

    obj = roxy.mcmc.OrderedNormal()
    rng_key = jax.random.PRNGKey(np.random.randint(1))
    obj.sample(rng_key, sample_shape=(5,))
    obj.log_prob(1)
    obj.cdf(1)
    obj.log_cdf(1)
    obj.icdf(0.5)
    obj.mean
    obj.variance
    
    # Make data for likelihood tests
    theta0 = [2, 0.5]
    nx = 20
    xerr = 0.1
    yerr = 0.5
    sig = 0.5
    np.random.seed(0)
    xtrue = np.linspace(0, 5, nx)
    ytrue = theta0[0] * xtrue + theta0[1]
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)
    f = ytrue
    fprime = jnp.ones(f.shape) * theta0[0]
    mu_gauss = 2.5
    w_gauss = 1.0
    Sxx = np.identity(nx) * xerr ** 2
    Sxy = np.zeros((nx,nx))
    Syy = np.identity(nx) * yerr ** 2
    G = np.identity(nx) * theta0[0]
    
    data = [xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss]
    all_obj = [roxy.mcmc.Likelihood_MNR(*data),
                roxy.mcmc.Likelihood_prof(*data[:-2]),
                roxy.mcmc.Likelihood_unif(*data[:-2]),
                ]
    data = [xobs, yobs, Sxx, Syy, Sxy, f, G, sig, mu_gauss, w_gauss]
    all_obj += [roxy.mcmc.Likelihood_MNR_MV(*data),
                roxy.mcmc.Likelihood_prof_MV(*data[:-2]),
                roxy.mcmc.Likelihood_unif_MV(*data[:-2]),
                ]
    data = [xobs, yobs, xerr, yerr, f, fprime, sig,
            [-10, 0.0], [1.0, 3.0], [0.7, 0.3]]
    all_obj += [roxy.mcmc.Likelihood_GMM(*data)]
    for obj in all_obj:
        try:
            obj.sample(rng_key, sample_shape=(5,))
        except NotImplementedError:
            pass

    return
    
    
def test_causality(monkeypatch):

    monkeypatch.setattr(plt, 'show', lambda: None)

    def my_fun(x, theta):
        return theta[0] * x + theta[1]

    def fun_inv(y, theta):
        return y / theta[0] - theta[1] / theta[0]
    
    param_names = ['A', 'B']
    theta0 = [0.4, 1.0]
    param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}

    reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

    nx = 100
    xerr = np.random.normal(1, 0.2, nx)
    yerr = np.random.normal(2, 0.2, nx)
    xerr[xerr<0]=0
    yerr[yerr<0]=0
    sig = 3.0

    np.random.seed(0)

    xtrue = np.random.uniform(0, 30, nx)
    ytrue = reg.value(xtrue, theta0)
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

    for criterion in ['spearman', 'pearson', 'hsic']:
        roxy.causality.assess_causality(my_fun, fun_inv, xobs, yobs, [xerr, yerr],
            param_names, theta0, param_prior, method='mnr',
            criterion=criterion, savename='causality.png', show=True)
    
    # Check for unknown criterion we get an error
    try:
        roxy.causality.assess_causality(my_fun, fun_inv, xobs, yobs, [xerr, yerr],
            param_names, theta0, param_prior, method='mnr',
            criterion='unknown_criterion', savename='causality.png', show=True)
    except NotImplementedError:
        pass
        
    # Check it works the other way around x <-> y
    roxy.causality.assess_causality(my_fun, fun_inv, yobs, xobs, [yerr, xerr],
            param_names, theta0, param_prior, method='mnr',
            criterion='hsic', savename='causality.png', show=True)
    
    # Now with covariance matrix
    Sxx = np.identity(nx) * xerr ** 2
    Sxy = np.zeros((nx,nx))
    Syx = np.zeros((nx,nx))
    Syy = np.identity(nx) * yerr ** 2
    Sigma = np.concatenate(
                [np.concatenate([Sxx, Sxy], axis=-1),
                np.concatenate([Syx, Syy], axis=-1)]
            )
    roxy.causality.assess_causality(my_fun, fun_inv, xobs, yobs, Sigma,
        param_names, theta0, param_prior, method='mnr', savename='causality.png',
        show=True, covmat=True)
        
    return
    
    
def test_nodiag_cov():

    def my_fun(x, theta):
        return theta[0] * x + theta[1]
        
    param_names = ['A', 'B']
    theta0 = [0.4, 1.0]
    param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}

    reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)
    
    np.random.seed(0)
        
    # Make true data
    nx = 20
    xtrue = np.random.uniform(0, 30, nx)
    ytrue = reg.value(xtrue, theta0)

    # Make a random covariance matrix which is non-diagonal
    # Sigma = A A^T since this is then a positive semi-definite, symmetric matrix
    Sigma = np.random.randn(2*nx, 2*nx) * 0.05
    Sigma = np.dot(Sigma, Sigma.transpose())
    
    # Generate observed data
    obs = np.random.multivariate_normal(
        np.concatenate([xtrue, ytrue]),
        Sigma)
    xobs = obs[:nx]
    yobs = obs[nx:]
    
    nwarm = 50
    nsamp = 50
    
    for method in ['unif', 'prof', 'mnr']:
        reg.optimise(param_names, xobs, yobs, Sigma, method='mnr', covmat=True)
        reg.mcmc(param_names, xobs, yobs, Sigma, nwarm, nsamp, method=method,
                covmat=True)
    return
    
def test_warnings():
    # Test that warnings are raised for incorrectly used likelihoods
    
    def my_fun(x, theta):
        return theta[0] * x + theta[1]
        
    param_names = ['A', 'B']
    theta0 = [0.4, 1.0]
    param_prior = {'A':[0, 5], 'B':[-2, 2], 'sig':[0, 3.0]}

    reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)
    
    np.random.seed(0)
        
    # Make true data
    nx = 20
    xtrue = np.random.uniform(0, 30, nx)
    ytrue = reg.value(xtrue, theta0)
    
    nwarm = 10
    nsamp = 10

    # ----------------------------------
    # Warnings with a float for errors
    
    xerr = 0.1
    yerr = 0.5
    sig = 0.5
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)
    
    for method in ['unif', 'prof']:
        with unittest.TestCase().assertWarns(UserWarning):
            reg.optimise(
                param_names, xobs, yobs, [xerr, yerr], method=method, covmat=False,
                infer_intrinsic=True)
        with unittest.TestCase().assertWarns(UserWarning):
            reg.mcmc(
                param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method,
                covmat=False, infer_intrinsic=True)
                
    for method in ['unif', 'mnr', 'gmm']:
        with unittest.TestCase().assertWarns(UserWarning):
            reg.optimise(
                param_names, xobs, yobs, [xerr, yerr], method=method, covmat=False,
                infer_intrinsic=False)
        with unittest.TestCase().assertWarns(UserWarning):
            reg.mcmc(
                param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method,
                covmat=False, infer_intrinsic=False)
                
    xerr = 0.
    
    for method in ['mnr', 'gmm']:
        for infer_intrinsic in [True, False]:
            with unittest.TestCase().assertWarns(UserWarning):
                reg.optimise(
                    param_names, xobs, yobs, [xerr, yerr], method=method, covmat=False,
                    infer_intrinsic=infer_intrinsic)
            with unittest.TestCase().assertWarns(UserWarning):
                reg.mcmc(
                    param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method,
                    covmat=False, infer_intrinsic=infer_intrinsic)
    
    # ----------------------------------
    # Warnings with a float for errors
    
    xerr = np.random.uniform(0, 1, len(xtrue))
    yerr = np.random.uniform(0, 1, len(ytrue))
    sig = 0.5
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)
    
    for method in ['unif', 'prof']:
        with unittest.TestCase().assertWarns(UserWarning):
            reg.optimise(
                param_names, xobs, yobs, [xerr, yerr], method=method, covmat=False,
                infer_intrinsic=True)
        with unittest.TestCase().assertWarns(UserWarning):
            reg.mcmc(
                param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method,
                covmat=False, infer_intrinsic=True)
                
    for method in ['unif', 'mnr', 'gmm']:
        with unittest.TestCase().assertWarns(UserWarning):
            reg.optimise(
                param_names, xobs, yobs, [xerr, yerr], method=method, covmat=False,
                infer_intrinsic=False)
        with unittest.TestCase().assertWarns(UserWarning):
            reg.mcmc(
                param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method,
                covmat=False, infer_intrinsic=False)
                
    xerr[:] = 0.
    
    for method in ['mnr', 'gmm']:
        for infer_intrinsic in [True, False]:
            with unittest.TestCase().assertWarns(UserWarning):
                reg.optimise(
                    param_names, xobs, yobs, [xerr, yerr], method=method, covmat=False,
                    infer_intrinsic=infer_intrinsic)
            with unittest.TestCase().assertWarns(UserWarning):
                reg.mcmc(
                    param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method,
                    covmat=False, infer_intrinsic=infer_intrinsic)
    
    # ----------------------------------
    # Warnings with a covariance matrix
    
    # Make a random covariance matrix which is non-diagonal
    # Sigma = A A^T since this is then a positive semi-definite, symmetric matrix
    Sigma = np.random.randn(2*nx, 2*nx) * 0.05
    Sigma = np.dot(Sigma, Sigma.transpose())
    
    # Generate observed data
    obs = np.random.multivariate_normal(
        np.concatenate([xtrue, ytrue]),
        Sigma)
    xobs = obs[:nx]
    yobs = obs[nx:]
    
    for method in ['unif', 'prof']:
        with unittest.TestCase().assertWarns(UserWarning):
            reg.optimise(
                param_names, xobs, yobs, Sigma, method=method, covmat=True,
                infer_intrinsic=True)
        with unittest.TestCase().assertWarns(UserWarning):
            reg.mcmc(
                param_names, xobs, yobs, Sigma, nwarm, nsamp, method=method,
                covmat=True, infer_intrinsic=True)
                
    for method in ['unif', 'mnr']:
        with unittest.TestCase().assertWarns(UserWarning):
            reg.optimise(
                param_names, xobs, yobs, Sigma, method=method, covmat=True,
                infer_intrinsic=False)
        with unittest.TestCase().assertWarns(UserWarning):
            reg.mcmc(
                param_names, xobs, yobs, Sigma, nwarm, nsamp, method=method,
                covmat=True, infer_intrinsic=False)
                
    Sigma[:nx,:nx] = 0.
    
    for method in ['mnr']:
        for infer_intrinsic in [True, False]:
            with unittest.TestCase().assertWarns(UserWarning):
                reg.optimise(
                    param_names, xobs, yobs, Sigma, method=method, covmat=True,
                    infer_intrinsic=infer_intrinsic)
            with unittest.TestCase().assertWarns(UserWarning):
                reg.mcmc(
                    param_names, xobs, yobs, Sigma, nwarm, nsamp, method=method,
                    covmat=True, infer_intrinsic=infer_intrinsic)

    return
