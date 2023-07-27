import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import numpy as np
import numpyro
import numpyro.distributions as dist
import warnings
from sklearn.mixture import GaussianMixture
from operator import attrgetter

import roxy.likelihoods
import roxy.mcmc

class RoxyRegressor():
    """
    Regressor class which handles optimisation and MCMC for ``roxy``. One can
    use this class to evaluate the function of interest and its derivative,
    optimise the parameters using and of the defined likelihoods and run an
    MCMC for these parameters.
    
    Args:
        :fun (callable): The function, f, to be considered by this regressor y = f(x, theta). The function must take two arguments, the first of which is the independent variable, the second of which are the parameters (as an array or list).
        :param_names (list): The list of parameter names, in the order which they are supplied to fun
        :param_default (list): The default valus of the parameters
        :param_prior (dict): The prior range for each of the parameters. The prior is assumed to be uniform in this range
    """

    def __init__(self, fun, param_names, param_default, param_prior):
        
        self.single_fun = fun
        self.single_gradfun = jax.grad(self.single_fun, argnums=0)
        self.fun = jax.vmap(self.single_fun, (0, None), 0)
        self.gradfun = jax.vmap(self.single_gradfun, (0, None), 0)
        
        self.param_names = param_names
        self.param_default = jnp.array(param_default)
        self.param_prior = param_prior
        
    def value(self, x, theta):
        """
        If we are fitting the function f(x, theta), this is f(x, theta) evaluated at (x, theta)
        
        Args:
            :x (jnp.ndarray): The x values
            :theta (jnp.ndarray): The parameter values
            
        Returns:
            :jnp.ndarray: f(x, theta) evaluated at (x, theta)
        """
        return self.fun(x, theta)
        
    def gradient(self, x, theta):
        """
        If we are fitting the function f(x, theta), this is df/dx evaluated at (x, theta)
                
        Args:
            :x (jnp.ndarray): The x values
            :theta (jnp.ndarray): The parameter values
            
        Returns:
            :jnp.ndarray: df/dx evaluated at (x, theta)
        """
        return self.gradfun(x, theta)
        
    def negloglike(self, theta, xobs, yobs, errors, sig=0., mu_gauss=0., w_gauss=1., weights_gauss=1., method='mnr', covmat=False):
        """
        Computes the negative log-likelihood under the assumption of
        an uncorrelated (correlated) Gaussian likelihood if covmat is False (True),
        using the likelihood specififed by 'method'.
        
        Args:
            :theta (jnp.ndarray): The parameters of the function to use
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving the error on the observed x and y values. Otherwise, this is the covariance matrix in the order (x, y)
            :sig (float, default=0.): The intrinsic scatter, which is added in quadrature with yerr
            :mu_gauss (float or jnp.ndarray, default=0.): The mean of the Gaussian prior on the true x positions (only used if method='mnr' or 'gmm'). If using 'mnr' and this is an array, only the first mean is used.
            :w_gauss (float or jnp.ndarray, default=1.): The standard deviation of the Gaussian prior on the true x positions (only used if method='mnr').
            :weights_gauss (float or jnp.ndarray, default=1.): The weights of the Gaussians in a GMM prior on the true x positions (only used if method='gmm').
            :method (str, default='mnr'): The name of the likelihood method to use ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more information
            :covmat (bool, default=False): This determines whether the errors argument is [xerr, yerr] (False) or a covariance matrix (True).
        """
        f = self.value(xobs, theta)
        if covmat:
            G = jax.jacrev(self.fun, argnums=0)(xobs, theta)
        else:
            fprime = self.gradient(xobs, theta)
            xerr, yerr = errors
        
        if sig < 0. or (method == 'mnr' and w_gauss < 0.):
            return np.nan
        
        if method == 'mnr':
            if covmat:
                return roxy.likelihoods.negloglike_mnr_mv(xobs, yobs, errors, f, G, sig, mu_gauss, w_gauss)
            else:
                return roxy.likelihoods.negloglike_mnr(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss)
        elif method == 'gmm':
            mu = jnp.array(mu_gauss)
            w = jnp.array(w_gauss)
            weights = jnp.array(weights_gauss)
            if covmat:
                raise NotImplementedError
            else:
                return roxy.likelihoods.negloglike_gmm(xobs, yobs, xerr, yerr, f, fprime, sig, mu, w, weights)
        elif method == 'unif':
            if covmat:
                return roxy.likelihoods.negloglike_unif_mv(xobs, yobs, errors, f, G, sig)
            else:
                return roxy.likelihoods.negloglike_unif(xobs, yobs, xerr, yerr, f, fprime, sig)
        elif method == 'prof':
            if covmat:
                return roxy.likelihoods.negloglike_prof_mv(xobs, yobs, errors, f, G, sig)
            else:
                return roxy.likelihoods.negloglike_prof(xobs, yobs, xerr, yerr, f, fprime, sig)
        else:
            raise NotImplementedError
            
    def get_param_index(self, params_to_opt, verbose=True):
        """
        If the function of interest if f(x, theta), find the index in theta for each
        of the parameters we wish to optimise
        
        Args:
            :params_to_opt (list): The names of the parameters we wish to optimise
            :verbose (bool, default=True): Whether to print the names and values of parameters which are not fitted
            
        Returns:
            :pidx (jnp.ndarray): The indices of the parameters to optimise
        """
        # Get indices of params to optimise
        pidx = [self.param_names.index(p) for p in params_to_opt if p in self.param_names]
        if len(pidx) != len(self.param_names) and verbose:
            print('\nNot optimising all parameters. Using defaults:')
            for pname, pdefault in zip(self.param_names, self.param_default):
                if pname not in params_to_opt:
                    print(f'{pname}:\t{pdefault}')
        return jnp.array(pidx)
            
    def optimise(self, params_to_opt, xobs, yobs, errors, method='mnr', infer_intrinsic=True, initial=None, ngauss=1, covmat=False, gmm_prior='hierarchical'):
        """
        Optimise the parameters of the function given some data, under the assumption of
        an uncorrelated (correlated) Gaussian likelihood if covmat is False (True),
        using the likelihood specififed by 'method'.
        
        Args:
            :params_to_opt (list): The names of the parameters we wish to optimise
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving the error on the observed x and y values. Otherwise, this is the covariance matrix in the order (x, y)
            :method (str, default='mnr'): The name of the likelihood method to use ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more information
            :infer_intrinsic (bool, default=True): Whether to infer the intrinsic scatter in the y direction
            :initial (jnp.ndarray, default=None): The starting point for the optimised. If None, a random value in the prior range is chosen
            :ngauss (int, default = 1): The number of Gaussians to use in the GMM prior. Only used if method='gmm'
            :covmat (bool, default=False): This determines whether the errors argument is [xerr, yerr] (False) or a covariance matrix (True).
            :gmm_prior (string, default='hierarchical'): If method='gmm', this decides what prior to put on the GMM componenents. If 'uniform', then the mean and widths have a uniform prior, and if 'hierarchical' mu and w^2 have a Normal and Inverse Gamma prior, respectively.
        
        Returns:
            :res (scipy.optimize._optimize.OptimizeResult): The result of the optimisation
            :param_names (list): List of parameter names in order of res.x
        """
    
        # Get indices of params to optimise
        pidx = self.get_param_index(params_to_opt)
        
        def fopt(theta):
        
            # Check prior
            for i, p in enumerate(params_to_opt):
                if theta[i] < self.param_prior[p][0] or theta[i] > self.param_prior[p][1]:
                    return np.inf
            if method == 'mnr':
                if theta[-2] < xobs.min() or theta[-1] > xobs.max():
                    return np.inf
                if theta[-1] < 0 or theta[-1] > 5 * xobs.std():
                    return np.inf
            if infer_intrinsic:
                if theta[len(pidx)] < self.param_prior['sig'][0] or theta[len(pidx)] > self.param_prior['sig'][1]:
                    return np.inf
                
            # Parameters of function
            t = self.param_default
            t = t.at[pidx].set(theta[:len(pidx)])
            
            # f(x) and f'(x) for these params
            f = self.value(xobs, t)
            fprime = self.gradient(xobs, t)
            
            # Intrinsic scatter
            if infer_intrinsic:
                sig = theta[len(pidx)]
            else:
                sig = 0.

            # Variable to store any prior knowledge
            nll = 0.
            
            # MNR parameters
            if method == 'mnr':
                mu_gauss = theta[-2]
                w_gauss = theta[-1]
                weights_gauss = 1.
            elif method == 'gmm':
                imin = len(params_to_opt)
                if infer_intrinsic:
                    imin += 1
                mu_gauss = theta[imin:imin+ngauss]
                w_gauss = theta[imin+ngauss:imin+2*ngauss]
                weights_gauss = np.zeros(ngauss)
                weights_gauss[:ngauss-1] = theta[imin+2*ngauss:imin+3*ngauss-1]
                weights_gauss[-1] = 1 - sum(weights_gauss)
                
                if jnp.any(weights_gauss > 1) or jnp.any(weights_gauss < 0):
                    return np.inf
                if jnp.any(w_gauss < 0):
                    return np.inf
                    
                if gmm_prior == 'uniform':
                    pass
                elif gmm_prior == 'hierarchical':
                    hyper_mu, hyper_w2, hyper_u2 = theta[imin+3*ngauss-1:]
                    if (hyper_w2 < 0) or (hyper_u2 < 0):
                        return np.inf
                    nll = (
                        0.5 * (-jnp.log(hyper_w2) + 3 * jnp.log(hyper_u2) + hyper_w2 / hyper_u2)
                        + jnp.sum(
                            0.5 * jnp.log(hyper_u2) + 3/2 * jnp.log(w_gauss) - 0.5 * jnp.log(hyper_w2) + jnp.log(2 * jnp.pi)
                            + (mu_gauss - hyper_mu) ** 2 / (2 * hyper_w2) + hyper_w2 / (2 * w_gauss ** 2)
                        )
                    )
                    
                else:
                    raise NotImplementedError
            else:
                mu_gauss, w_gauss, weights_gauss = None, None, None

            return nll + self.negloglike(t, xobs, yobs, errors, sig=sig, mu_gauss=mu_gauss, w_gauss=w_gauss, weights_gauss=weights_gauss, method=method, covmat=covmat)
        
        # Get initial guess
        if initial is None:
            initial = [np.random.uniform(*self.param_prior[p]) for p in params_to_opt]
            if infer_intrinsic:
                initial = initial + [np.random.uniform(*self.param_prior['sig'])]
            if method == 'mnr':
                initial = initial + [np.mean(xobs), np.std(xobs)]
            elif method == 'gmm':
                gm = GaussianMixture(n_components=ngauss, random_state=0).fit(xobs.reshape(-1,1))
                gm_means = np.atleast_1d(np.squeeze(gm.means_))
                gm_ws = np.sqrt(np.atleast_1d(np.squeeze(gm.covariances_)))
                gm_weights = np.atleast_1d(np.squeeze(gm.weights_))
                idx = np.argsort(gm_means)
                if gmm_prior == 'uniform':
                    initial = jnp.array(
                        initial
                        + list(gm_means[idx])
                        + list(gm_ws[idx])
                        + list((gm_weights[idx])[:ngauss - 1])
                    )
                elif gmm_prior == 'hierarchical':
                    initial = jnp.array(
                        initial
                        + list(gm_means[idx])
                        + list(gm_ws[idx])
                        + list((gm_weights[idx])[:ngauss - 1])
                        + [gm_means[idx[0]], gm_ws[idx[0]]**2, gm_ws[idx[0]]**2/3]
                    )
                else:
                    raise NotImplementedError
            
        res = minimize(fopt, initial, method="Nelder-Mead")
        
        # Print results
        print('\nOptimisation Results:')
        param_names = []
        for p, val in zip(params_to_opt, res.x):
            print(f'{p}:\t{val}')
            param_names.append(p)
        if infer_intrinsic:
            print(f'sig:\t{res.x[len(params_to_opt)]}')
            param_names.append('sig')
        if method == 'mnr':
            print(f'mu_gauss:\t{res.x[-2]}')
            print(f'w_gauss:\t{res.x[-1]}')
            param_names.append('mu_gauss')
            param_names.append('w_gauss')
        elif method == 'gmm':
            imin = len(params_to_opt)
            if infer_intrinsic:
                imin += 1
            for i in range(ngauss):
                print(f'mu_gauss_{i}:\t{res.x[imin+i]}')
                param_names.append(f'mu_gauss_{i}')
            for i in range(ngauss):
                print(f'w_gauss_{i}:\t{res.x[imin+ngauss+i]}')
                param_names.append(f'w_gauss_{i}')
            for i in range(ngauss-1):
                print(f'weight_gauss_{i}:\t{res.x[imin+2*ngauss+i]}')
                param_names.append(f'weight_gauss_{i}')
            if gmm_prior == 'hierarchical':
                print(f'hyper_mu:\t{res.x[imin+3*ngauss-1]}')
                param_names.append(f'hyper_mu')
                print(f'hyper_u2:\t{res.x[imin+3*ngauss]}')
                param_names.append(f'hyper_w2')
                print(f'hyper_w2:\t{res.x[imin+3*ngauss+1]}')
                param_names.append(f'hyper_u2')
        
        return res, param_names

    def mcmc(self, params_to_opt, xobs, yobs, errors, nwarm, nsamp, method='mnr', ngauss=1., infer_intrinsic=True, progress_bar=True, covmat=False, gmm_prior='hierarchical', seed=1234):
        """
        Run an MCMC using the NUTS sampler of ``numpyro`` for the parameters of the
        function given some data, under the assumption of an uncorrelated Gaussian likelihood,
        using the likelihood specififed by 'method'.
        
        Args:
            :params_to_opt (list): The names of the parameters we wish to optimise
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving the error on the observed x and y values. Otherwise, this is the covariance matrix in the order (x, y)
            :nwarm (int): The number of warmup steps to use in the MCMC
            :nsamp (int): The number of samples to obtain in the MCMC
            :method (str, default='mnr'): The name of the likelihood method to use ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more information.
            :ngauss (int, default = 1): The number of Gaussians to use in the GMM prior. Only used if method='gmm'
            :infer_intrinsic (bool, default=True): Whether to infer the intrinsic scatter in the y direction
            :progress_bar (bool, default=True): Whether to display a progress bar for the MCMC
            :covmat (bool, default=False): This determines whether the errors argument is [xerr, yerr] (False) or a covariance matrix (True).
            :gmm_prior (string, default='hierarchical'): If method='gmm', this decides what prior to put on the GMM componenents. If 'uniform', then the mean and widths have a uniform prior, and if 'hierarchical' mu and w^2 have a Normal and Inverse Gamma prior, respectively.
            :seed (int, default=1234): The seed to use when initialising the sampler
            
        Returns:
            :samples (dict): The MCMC samples, where the keys are the parameter names and values are ndarrays of the samples
        """

        pidx = self.get_param_index(params_to_opt, verbose=False)
        
        if covmat:
            nx = len(xobs)
            Sxx = errors[:nx, :nx]
            Sxy = errors[:nx, nx:]
            Syy = errors[nx:, nx:]
        else:
            xerr, yerr = errors
                
        def model():
        
            # Parameters of function
            theta = [numpyro.sample(p, dist.Uniform(*self.param_prior[p])) for p in params_to_opt]
            t = self.param_default
            t = t.at[pidx].set(theta[:len(pidx)])
            
            # f(x) and f'(x) for these params
            f = self.value(xobs, t)
            if covmat:
                G = jax.jacrev(self.fun, argnums=0)(xobs, t)
            else:
                fprime = self.gradient(xobs, t)
            
            # Intrinsic scatter
            if infer_intrinsic:
                sig = numpyro.sample("sig", dist.Uniform(*self.param_prior['sig']))
            else:
                sig = 0.
                
            # MNR parameters
            if method == 'mnr':
                mu_gauss = numpyro.sample("mu_gauss", dist.Uniform(xobs.min(), xobs.max()))
                w_gauss = numpyro.sample("w_gauss", dist.Uniform(0., 5*jnp.std(xobs)))
            elif method == 'gmm':
                if gmm_prior == 'uniform':
                    all_mu_gauss = numpyro.sample("mu_gauss", dist.ImproperUniform(dist.constraints.ordered_vector, (), (ngauss,)))
                    all_w_gauss = numpyro.sample("w_gauss", dist.Uniform(0., 5*jnp.std(xobs)), sample_shape=(ngauss,))
                    all_weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(ngauss)))
                elif gmm_prior == 'hierarchical':
                    hyper_mu = numpyro.sample("hyper_mu", dist.ImproperUniform(dist.constraints.real, (), event_shape=()))
                    hyper_w2 = numpyro.sample("hyper_w2", dist.ImproperUniform(dist.constraints.positive, (), event_shape=()))
                    hyper_u2 = numpyro.sample("hyper_u2", dist.InverseGamma(1/2, hyper_w2/2))
                    all_mu_gauss = numpyro.sample("mu_gauss", roxy.mcmc.OrderedNormal(hyper_mu, jnp.sqrt(hyper_u2)), sample_shape=(ngauss,))
                    all_w_gauss = jnp.sqrt(numpyro.sample("w_gauss", dist.InverseGamma(1/2, hyper_w2/2), sample_shape=(ngauss,)))
                    all_weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(ngauss)))
                else:
                    raise NotImplementedError

            # Sample
            if method == 'mnr':
                if covmat:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_MNR_MV(xobs, yobs, Sxx, Syy, Sxy, f, G, sig, mu_gauss, w_gauss),
                        obs=yobs,
                    )
                else:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_MNR(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss),
                        obs=yobs,
                    )
            elif method == 'unif':
                if covmat:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_unif_MV(xobs, yobs, Sxx, Syy, Sxy, f, G, sig),
                        obs=yobs,
                    )
                else:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_unif(xobs, yobs, xerr, yerr, f, fprime, sig),
                        obs=yobs,
                    )
            elif method == 'prof':
                if covmat:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_prof_MV(xobs, yobs, Sxx, Syy, Sxy, f, G, sig),
                        obs=yobs,
                    )
                else:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_prof(xobs, yobs, xerr, yerr, f, fprime, sig),
                        obs=yobs,
                    )
            elif method == 'gmm':
                if covmat:
                    raise NotImplementedError
                else:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_GMM(xobs, yobs, xerr, yerr, f, fprime, sig, all_mu_gauss, all_w_gauss, all_weights),
                        obs=yobs,
                    )
            else:
                raise NotImplementedError
             
        rng_key = jax.random.PRNGKey(np.random.randint(seed))
        rng_key, rng_key_ = jax.random.split(rng_key)
        
        try:
            vals, param_names = self.optimise(params_to_opt, xobs, yobs, errors, method=method, infer_intrinsic=infer_intrinsic, ngauss=ngauss, covmat=covmat, gmm_prior=gmm_prior)
            vals = vals.x
            init = {k:v for k,v in zip(param_names, vals)}
            if 'mu_gauss_0' in param_names:
                init_mu = [0.] * ngauss
                init_w = [0.] * ngauss
                init_weight = [0.] * ngauss
                for i in range(ngauss):
                    init_mu[i] = init[f'mu_gauss_{i}']
                    init_w[i] = init[f'w_gauss_{i}']
                    init.pop(f'mu_gauss_{i}')
                    init.pop(f'w_gauss_{i}')
                for i in range(ngauss - 1):
                    init_weight[i] = init[f'weight_gauss_{i}']
                    init.pop(f'weight_gauss_{i}')
                init_weight[-1] = 1. - sum(init_weight)
                init['mu_gauss'] = jnp.array(init_mu)
                if gmm_prior == 'uniform':
                    init['w_gauss'] = jnp.array(init_w)
                elif gmm_prior == 'hierarchical':
                    init['w_gauss'] = jnp.array(init_w) ** 2
                init['weight_gauss'] = jnp.array(init_weight)
                idx = jnp.argsort(init['mu_gauss'])
                init['mu_gauss'] = init['mu_gauss'][idx]
                init['w_gauss'] = init['w_gauss'][idx]
                init['weight_gauss'] = init['weight_gauss'][idx]
            kernel = numpyro.infer.NUTS(model, init_strategy=numpyro.infer.initialization.init_to_value(values=init))
            print('\nRunning MCMC')
            sampler = numpyro.infer.MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, progress_bar=progress_bar)
            sampler.run(rng_key_)
        except:
            print('\nCould not init to optimised values')
            kernel = numpyro.infer.NUTS(model)
            print('\nRunning MCMC')
            sampler = numpyro.infer.MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, progress_bar=progress_bar)
            sampler.run(rng_key_)

        samples = sampler.get_samples()
        
        # We actually samples w2 if gmm_prior = 'hierarchical', so correct for this
        if method == 'gmm' and gmm_prior == 'hierarchical':
            samples['w_gauss'] = jnp.sqrt(samples['w_gauss'])
            
        # Print summary
        sites = samples
        if isinstance(samples, dict):
            state_sample_field = attrgetter(sampler._sample_field)(sampler._last_state)
            if isinstance(state_sample_field, dict):
                sites = {
                    k: jnp.expand_dims(v, axis=0)
                    for k, v in samples.items()
                    if k in state_sample_field
                }
        numpyro.diagnostics.print_summary(sites, prob=0.95)
        extra_fields = sampler.get_extra_fields()
        if "diverging" in extra_fields:
            print(
                "Number of divergences: {}".format(jnp.sum(extra_fields["diverging"]))
            )
        
        # Raise warning if too few effective samples
        neff = np.zeros(len(samples))
        for i, (k, v) in enumerate(samples.items()):
            x = jnp.expand_dims(v, axis=0)
            try:
                neff[i] = numpyro.diagnostics.effective_sample_size(x)
            except:
                neff[i] = min(numpyro.diagnostics.effective_sample_size(x))
        m = neff < 100
        if m.sum() > 0:
            bad_keys = [k for i,k in enumerate(samples.keys()) if m[i]]
            warnings.warn('Fewer than 100 effective samples for parameters: ' + ', '.join(bad_keys), category=Warning, stacklevel=2)
            
        # Raise warning if the peak of the posterior is too close to the edge of the prior
        bad_keys = []
        for p in params_to_opt:
            counts, _ = np.histogram(samples[p], np.linspace(self.param_prior[p][0], self.param_prior[p][1], 30))
            if (np.argmax(counts) < 2) or (np.argmax(counts) > 27):
                bad_keys.append(p)
        if len(bad_keys) > 0:
            warnings.warn('Posterior near edge of prior for parameters: ' + ', '.join(bad_keys), category=Warning, stacklevel=2)
        
        return samples
        
    def find_best_gmm(self, params_to_opt, xobs, yobs, xerr, yerr, max_ngauss, best_metric='BIC', infer_intrinsic=True, nwarm=100, nsamp=100, gmm_prior='hierarchical', seed=1234):
        """
        Find the number of Gaussians to use in a Gaussian Mixture Model
        hyper-prior on the true x values, accoridng to some metric.
        We first run a MCMC to get an initial guess for the maximum likelihood
        point, and we then an optimsier from this point to get a better
        estimate for this. Since we do not need good MCMC convergence for this,
        small values of nwarm and nsamp can be used.
        
        Args:
            :params_to_opt (list): The names of the parameters we wish to optimise
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :xerr (jnp.ndarray): The error on the observed x values
            :yerr (jnp.ndarray): The error on the observed y values
            :max_ngauss (int): The maximum number of Gaussians to consider
            :best_metric (str): Metric to use to compare fits (supported: AIC and BIC)
            :infer_intrinsic (bool, default=True): Whether to infer the intrinsic scatter in the y direction
            :nwarm (int, default=100): The number of warmup steps to use in the MCMC
            :nsamp (int, default=100): The number of samples to obtain in the MCMC
            :gmm_prior (string, default='hierarchical'): If method='gmm', this decides what prior to put on the GMM componenents. If 'uniform', then the mean and widths have a uniform prior, and if 'hierarchical' mu and w^2 have a Normal and Inverse Gamma prior, respectively.
            :seed (int, default=1234): The seed to use when initialising the sampler
            
        Returns:
            :ngauss (int): The best number of Gaussians to use according to the metric
        
        """
    
        npar = np.empty(max_ngauss)
        negloglike = np.empty(max_ngauss)
    
        for ngauss in range(1, max_ngauss+1):
            
            # First run a MCMC to get a guess at the peak, catching the low neff warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = self.mcmc(params_to_opt,
                            xobs,
                            yobs,
                            [xerr, yerr],
                            nwarm,
                            nsamp,
                            method='gmm',
                            ngauss=ngauss,
                            infer_intrinsic=infer_intrinsic,
                            progress_bar=True,
                            gmm_prior=gmm_prior,
                            seed=seed
                )
            labels, samples = roxy.mcmc.samples_to_array(samples)
            labels = list(labels)
            
            # Now put in order expected by optimisers
            param_idx = [i for i, k in enumerate(labels) if not (k.startswith('weights') or k.startswith('mu_gauss') or k.startswith('w_gauss') or k.startswith('sig') or k.startswith('hierarchical'))]
            if infer_intrinsic:
                param_idx = param_idx + [labels.index('sig')]
            param_idx = param_idx + [labels.index(f'mu_gauss_{i}') for i in range(ngauss)]
            param_idx = param_idx + [labels.index(f'w_gauss_{i}') for i in range(ngauss)]
            param_idx = param_idx + [labels.index(f'weights_{i}') for i in range(ngauss-1)]
            if gmm_prior == 'hierarchical':
                param_idx = param_idx + [labels.index('hyper_mu'), labels.index('hyper_w2'), labels.index('hyper_u2')]
            param_names = [labels[i] for i in param_idx]
            
            # Extract medians
            initial = jnp.median(samples[:,param_idx], axis=0)
            # Run new optimiser
            res, _ = self.optimise(params_to_opt,
                        xobs,
                        yobs,
                        [xerr, yerr],
                        method='gmm',
                        infer_intrinsic=infer_intrinsic,
                        initial=initial,
                        ngauss=ngauss,
                        gmm_prior=gmm_prior
            )
            npar[ngauss-1] = len(initial)
            negloglike[ngauss-1] = res.fun

        if best_metric == 'AIC':
            metric = 2 * negloglike + 2 * npar
        elif best_metric == 'BIC':
            metric = 2 * negloglike + npar * jnp.log(len(xobs))
        else:
            raise NotImplementedError
        
        ngauss = np.argmin(metric) + 1
        print(f'\nBest ngauss according to {best_metric}:', ngauss)
        metric -= np.amin(metric)
        for i, m in enumerate(metric):
            print(i+1, m)

        return ngauss
