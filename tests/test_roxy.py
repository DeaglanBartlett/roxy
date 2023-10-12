import numpy as np
from roxy.regressor import RoxyRegressor
import roxy.plotting
import roxy.mcmc
import jax.random
import jax.numpy as jnp
import roxy.likelihoods

def test_example_standard():

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
    xtrue = np.linspace(0, 5, nx)
    ytrue = reg.value(xtrue, theta0)
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig ** 2)

    nwarm, nsamp = 70, 500
    samples = reg.mcmc(param_names, xobs, yobs, [xerr, yerr],
                nwarm, nsamp, method='mnr')

    roxy.plotting.trace_plot(samples, to_plot='all', savename=None, show=False)
    roxy.plotting.posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr,
        show=False, savename=None)
        
    # Just plot some variables
    roxy.plotting.triangle_plot(samples, to_plot=['A', 'B'], module='getdist',
        param_prior=param_prior, savename=None, show=False)
    roxy.plotting.trace_plot(samples, to_plot=['A', 'B'], savename=None, show=False)
    
    # Param prior checks
    roxy.plotting.triangle_plot(samples, to_plot=['A', 'B'], module='getdist',
        param_prior=None, savename=None, show=False)
    param_prior['sig'] = [0, None]
    roxy.plotting.triangle_plot(samples, to_plot=['A', 'B'], module='getdist',
        param_prior=param_prior, savename=None, show=False)
            
    # Check labels
    roxy.plotting.triangle_plot(samples, to_plot=['A', 'B'], module='getdist',
        param_prior=param_prior, savename=None, show=False, labels={'A':'A', 'B':'B'})
        
    for mod in ['getdist', 'corner']:
        roxy.plotting.triangle_plot(samples, to_plot='all', module=mod,
        param_prior=param_prior, savename=None, show=False)
        
    try:
        roxy.plotting.triangle_plot(samples, to_plot='all', module='badmodule',
            param_prior=param_prior, savename=None, show=False)
    except NotImplementedError:
        pass
        
    # Test biases
    roxy.mcmc.compute_bias(samples, truths, verbose=True)
    
    # A few likelihood checks
    for Ai in [theta0[0], [theta0[0]]]:
        roxy.likelihoods.negloglike_mnr(xobs, yobs, xerr, yerr, ytrue,
            Ai, sig, 2.5, 1.0)
        roxy.likelihoods.negloglike_gmm(xobs, yobs, xerr, yerr, ytrue,
            Ai, sig, [2.5], [1.0], [1.0])
        roxy.likelihoods.negloglike_unif(xobs, yobs, xerr, yerr, ytrue,
            Ai, sig)
            
    # Test regressor
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

    max_ngauss = 3
    np.random.seed(42)
    ngauss = reg.find_best_gmm(param_names, xobs, yobs, xerr, yerr, max_ngauss,
                best_metric='BIC', nwarm=100, nsamp=100, gmm_prior='uniform')
    assert ngauss == 2, "Did not find 2 Gaussians for case which clearly needs 2"
                
    for criterion in ['AIC', 'BIC']:
        reg.compute_information_criterion(criterion, param_names, xobs, yobs,
            [xerr, yerr], ngauss=1)
    try:
        reg.compute_information_criterion('DIC', param_names, xobs, yobs,
            [xerr, yerr], ngauss=1)
    except NotImplementedError:
        pass
                
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

if __name__ == "__main__":
    test_example_standard()
    test_example_gmm()
    test_different_likes()
    test_mcmc_classes()
