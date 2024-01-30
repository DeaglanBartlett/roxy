import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import warnings
from sklearn.mixture import GaussianMixture
from operator import attrgetter
from jaxopt import ScipyBoundedMinimize

import roxy.likelihoods
import roxy.mcmc

class OptResult(object):
    """
    Class to make the output of a jaxopt optimisation appear like
    a scipy.optimize._optimize.OptimizeResult
    
    Args:
        :res (ScipyMinimizeInfo): The result of a jaxopt minimisation
    """
    
    def __init__(self, res):
        self.x = res.params
        self.success = res.state.success
        self.status = res.state.status
        self.message = None
        self.fun = float(res.state.fun_val)
        self.jac = None
        self.hess = None
        self.hess_inv = res.state.hess_inv
        self.nfev = None
        self.njev = None
        self.nhev = None
        self.nit = res.state.iter_num
        self.maxcv = None

class RoxyRegressor():
    """
    Regressor class which handles optimisation and MCMC for ``roxy``. One can
    use this class to evaluate the function of interest and its derivative,
    optimise the parameters using and of the defined likelihoods and run an
    MCMC for these parameters.
    
    Args:
        :fun (callable): The function, f, to be considered by this regressor
            y = f(x, theta). The function must take two arguments, the first of which
            is the independent variable, the second of which are the parameters (as an
            array or list).
        :param_names (list): The list of parameter names, in the order which they are
            supplied to fun
        :param_default (list): The default valus of the parameters
        :param_prior (dict): The prior range for each of the parameters. The prior is
            assumed to be uniform in this range. If either entry is None in the prior,
            then an infinite uniform prior is assumed.
    """

    def __init__(self, fun, param_names, param_default, param_prior):
        
        self.single_fun = fun
        self.single_gradfun = jax.grad(self.single_fun, argnums=0)
        self.single_secondgradfun = jax.grad(self.single_gradfun, argnums=0)
        self.fun = jax.vmap(self.single_fun, (0, None), 0)
        self.gradfun = jax.vmap(self.single_gradfun, (0, None), 0)
        self.secondgradfun = jax.vmap(self.single_secondgradfun, (0, None), 0)
        
        self.param_names = param_names
        self.param_default = jnp.array(param_default).astype(jnp.float32)
        self.param_prior = param_prior
        
    def value(self, x, theta):
        """
        If we are fitting the function f(x, theta), this is f(x, theta) evaluated at
        (x, theta)
        
        Args:
            :x (jnp.ndarray): The x values
            :theta (jnp.ndarray): The parameter values
            
        Returns:
            :jnp.ndarray: f(x, theta) evaluated at (x, theta)
        """
        return self.fun(x, theta)
        
    def gradient(self, x, theta):
        """
        If we are fitting the function f(x, theta), this is df/dx evaluated at
        (x, theta)
                
        Args:
            :x (jnp.ndarray): The x values
            :theta (jnp.ndarray): The parameter values
            
        Returns:
            :jnp.ndarray: df/dx evaluated at (x, theta)
        """
        return self.gradfun(x, theta)
        
    def second_derivative(self, x, theta):
        """
        If we are fitting the function f(x, theta), this is d^2f/dx^2 evaluated at
        (x, theta)
                
        Args:
            :x (jnp.ndarray): The x values
            :theta (jnp.ndarray): The parameter values
            
        Returns:
            :jnp.ndarray: d^2f/dx^2 evaluated at (x, theta)
        """
        return self.secondgradfun(x, theta)
        
    def negloglike(self, theta, xobs, yobs, errors, sig=0., mu_gauss=0., w_gauss=1.,
        weights_gauss=1., method='mnr', covmat=False, test_prior=True,
        include_logdet=True):
        """
        Computes the negative log-likelihood under the assumption of
        an uncorrelated (correlated) Gaussian likelihood if covmat is False (True),
        using the likelihood specififed by 'method'.
        
        Args:
            :theta (jnp.ndarray): The parameters of the function to use
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving
                the error on the observed x and y values. Otherwise, this is the
                covariance matrix in the order (x, y)
            :sig (float, default=0.): The intrinsic scatter, which is added in
                quadrature with yerr
            :mu_gauss (float or jnp.ndarray, default=0.): The mean of the Gaussian
                prior on the true x positions (only used if method='mnr' or 'gmm').
                If using 'mnr' and this is an array, only the first mean is used.
            :w_gauss (float or jnp.ndarray, default=1.): The standard deviation of the
                Gaussian prior on the true x positions (only used if method='mnr').
            :weights_gauss (float or jnp.ndarray, default=1.): The weights of the
                Gaussians in a GMM prior on the true x positions (only used if
                method='gmm').
            :method (str, default='mnr'): The name of the likelihood method to use
                ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more
                information
            :covmat (bool, default=False): This determines whether the errors argument
                is [xerr, yerr] (False) or a covariance matrix (True).
            :test_prior (bool, default=True): Whether to test sigma >= 0 and Gaussians
                weights >= 0
            :include_logdet (bool, default=True): For the method 'prof', whether to
                include the normalisation term in the likelihood proportional
                to log(det(S))
        """
        f = self.value(xobs, theta)
        if covmat:
            G = jax.jacrev(self.fun, argnums=0)(xobs, theta)
        else:
            fprime = self.gradient(xobs, theta)
            xerr, yerr = errors
        
        if test_prior:
            if sig < 0. or (method == 'mnr' and w_gauss < 0.):
                return np.nan
        
        if method == 'mnr':
            if covmat:
                return roxy.likelihoods.negloglike_mnr_mv(xobs, yobs, errors, f, G, sig,
                        mu_gauss, w_gauss)
            else:
                return roxy.likelihoods.negloglike_mnr(xobs, yobs, xerr, yerr, f,
                        fprime, sig, mu_gauss, w_gauss)
        elif method == 'gmm':
            mu = jnp.array(mu_gauss)
            w = jnp.array(w_gauss)
            weights = jnp.array(weights_gauss)
            if covmat:
                raise NotImplementedError
            else:
                return roxy.likelihoods.negloglike_gmm(xobs, yobs, xerr, yerr, f,
                        fprime, sig, mu, w, weights)
        elif method == 'unif':
            if covmat:
                return roxy.likelihoods.negloglike_unif_mv(xobs, yobs, errors, f, G,
                        sig)
            else:
                return roxy.likelihoods.negloglike_unif(xobs, yobs, xerr, yerr, f,
                        fprime, sig)
        elif method == 'prof':
            if covmat:
                return roxy.likelihoods.negloglike_prof_mv(xobs, yobs, errors, f, G,
                        sig, include_logdet=include_logdet)
            else:
                return roxy.likelihoods.negloglike_prof(xobs, yobs, xerr, yerr, f,
                        fprime, sig, include_logdet=include_logdet)
        else:
            raise NotImplementedError
            
    def get_param_index(self, params_to_opt, verbose=True):
        """
        If the function of interest if f(x, theta), find the index in theta for each
        of the parameters we wish to optimise
        
        Args:
            :params_to_opt (list): The names of the parameters we wish to optimise
            :verbose (bool, default=True): Whether to print the names and values of
                parameters which are not fitted
            
        Returns:
            :pidx (jnp.ndarray): The indices of the parameters to optimise
        """
        # Get indices of params to optimise
        pidx = [self.param_names.index(p) for p in params_to_opt if
            p in self.param_names]
        if len(pidx) != len(self.param_names) and verbose:
            print('\nNot optimising all parameters. Using defaults:')
            for pname, pdefault in zip(self.param_names, self.param_default):
                if pname not in params_to_opt:
                    print(f'{pname}:\t{pdefault}')
        return jnp.array(pidx)
            
    def optimise(self, params_to_opt, xobs, yobs, errors, method='mnr',
            infer_intrinsic=True, initial=None, ngauss=1, covmat=False,
            gmm_prior='hierarchical', include_logdet=True, verbose=True):
        """
        Optimise the parameters of the function given some data, under the assumption of
        an uncorrelated (correlated) Gaussian likelihood if covmat is False (True),
        using the likelihood specififed by 'method'.
        
        Args:
            :params_to_opt (list): The names of the parameters we wish to optimise
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving
                the error on the observed x and y values. Otherwise, this is the
                covariance matrix in the order (x, y)
            :method (str, default='mnr'): The name of the likelihood method to use
                ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more
                information
            :infer_intrinsic (bool, default=True): Whether to infer the intrinsic
                scatter in the y direction
            :initial (jnp.ndarray, default=None): The starting point for the optimised.
                If None, a random value in the prior range is chosen
            :ngauss (int, default = 1): The number of Gaussians to use in the GMM prior.
                Only used if method='gmm'
            :covmat (bool, default=False): This determines whether the errors argument
                is [xerr, yerr] (False) or a covariance matrix (True).
            :gmm_prior (string, default='hierarchical'): If method='gmm', this decides
                what prior to put on the GMM componenents. If 'uniform', then the mean
                and widths have a uniform prior, and if 'hierarchical' mu and w^2 have
                a Normal and Inverse Gamma prior, respectively.
            :include_logdet (bool, default=True): For the method 'prof', whether to
                include the normalisation term in the likelihood proportional
                to log(det(S))
            :verbose (bool, default=True): Whether to print progress or not
            
        
        Returns:
            :res (OptResult): The result of the optimisation
            :param_names (list): List of parameter names in order of res.params
        """
    
        # Get indices of params to optimise
        pidx = self.get_param_index(params_to_opt)
        
        def fopt(theta):

            bad_run = False
                
            # Parameters of function
            t = self.param_default
            t = t.at[pidx].set(theta[:len(pidx)])
            
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
                weights_gauss = jnp.zeros(ngauss)
                weights_gauss = weights_gauss.at[:ngauss-1].set(
                                theta[imin+2*ngauss:imin+3*ngauss-1])
                weights_gauss = weights_gauss.at[-1].set(1 - jnp.sum(weights_gauss))
                
                bad_run = jnp.any(weights_gauss > 1) | jnp.any(weights_gauss < 0)
                    
                if gmm_prior == 'uniform':
                    pass
                elif gmm_prior == 'hierarchical':
                    hyper_mu, hyper_w2, hyper_u2 = theta[imin+3*ngauss-1:]
                    nll = (
                        0.5 * (-jnp.log(hyper_w2) + 3 * jnp.log(hyper_u2)
                        + hyper_w2 / hyper_u2)
                        + jnp.sum(
                            0.5 * jnp.log(hyper_u2) + 3/2 * jnp.log(w_gauss)
                            - 0.5 * jnp.log(hyper_w2) + jnp.log(2 * jnp.pi)
                            + (mu_gauss - hyper_mu) ** 2 / (2 * hyper_w2)
                            + hyper_w2 / (2 * w_gauss ** 2)
                        )
                    )
            else:
                mu_gauss, w_gauss, weights_gauss = None, None, None

            ll = nll + self.negloglike(t, xobs, yobs, errors, sig=sig,
                mu_gauss=mu_gauss, w_gauss=w_gauss, weights_gauss=weights_gauss,
                method=method, covmat=covmat, test_prior=False,
                include_logdet=include_logdet)
            ll = jnp.where(bad_run, np.inf, ll)
            
            return ll
        
        # Get initial guess
        if initial is None:
            initial = [None] * len(params_to_opt)
            for i, p in enumerate(params_to_opt):
                if ((self.param_prior[p][0] is not None) and
                    (self.param_prior[p][1] is not None)):
                    initial[i] = np.random.uniform(*self.param_prior[p])
                else:
                    initial[i] = np.random.uniform(0, 1)
            if infer_intrinsic:
                if ((self.param_prior['sig'][0] is not None) and
                    (self.param_prior['sig'][1] is not None)):
                    initial = initial + [np.random.uniform(*self.param_prior['sig'])]
                else:
                    initial = initial + [np.random.uniform(0, 1)]
            if method == 'mnr':
                initial = initial + [np.mean(xobs), np.std(xobs)]
            elif method == 'gmm':
                gm = GaussianMixture(n_components=ngauss, random_state=0).fit(
                    xobs.reshape(-1,1))
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
            
        initial = jnp.array(initial)
        lbfgsb = ScipyBoundedMinimize(fun=fopt, method="l-bfgs-b")
        lower_bounds = jnp.ones_like(initial) * (-jnp.inf)
        upper_bounds = jnp.ones_like(initial) * jnp.inf
        for i, p in enumerate(params_to_opt):
            if ((self.param_prior[p][0] is not None) and
                (self.param_prior[p][1] is not None)):
                lower_bounds = lower_bounds.at[i].set(self.param_prior[p][0])
                upper_bounds = upper_bounds.at[i].set(self.param_prior[p][1])
        if infer_intrinsic:
            if ((self.param_prior['sig'][0] is not None) and
                (self.param_prior['sig'][1] is not None)):
                lower_bounds = lower_bounds.at[len(pidx)].set(
                    self.param_prior['sig'][0])
                upper_bounds = upper_bounds.at[len(pidx)].set(
                    self.param_prior['sig'][1])
            else:
                lower_bounds = lower_bounds.at[len(pidx)].set(0)
        if method == 'gmm':
            imin = len(params_to_opt)
            if infer_intrinsic:
                imin += 1
            # Widths
            lower_bounds = lower_bounds.at[imin+ngauss:imin+2*ngauss].set(0.)
            # Weights
            lower_bounds = lower_bounds.at[imin+2*ngauss:imin+3*ngauss-1].set(0.)
            upper_bounds = upper_bounds.at[imin+2*ngauss:imin+3*ngauss-1].set(1.)
            # Hierarchical params
            lower_bounds = lower_bounds.at[imin+3*ngauss:].set(0.)
        res = lbfgsb.run(initial, bounds=(lower_bounds, upper_bounds))
        res = OptResult(res)
        
        # Print results
        if verbose:
            print('\nOptimisation Results:')
        param_names = []
        for p, val in zip(params_to_opt, res.x):
            if verbose:
                print(f'{p}:\t{val}')
            param_names.append(p)
        if infer_intrinsic:
            if verbose:
                print(f'sig:\t{res.x[len(params_to_opt)]}')
            param_names.append('sig')
        if method == 'mnr':
            if verbose:
                print(f'mu_gauss:\t{res.x[-2]}')
                print(f'w_gauss:\t{res.x[-1]}')
            param_names.append('mu_gauss')
            param_names.append('w_gauss')
        elif method == 'gmm':
            imin = len(params_to_opt)
            if infer_intrinsic:
                imin += 1
            for i in range(ngauss):
                if verbose:
                    print(f'mu_gauss_{i}:\t{res.x[imin+i]}')
                param_names.append(f'mu_gauss_{i}')
            for i in range(ngauss):
                if verbose:
                    print(f'w_gauss_{i}:\t{res.x[imin+ngauss+i]}')
                param_names.append(f'w_gauss_{i}')
            for i in range(ngauss-1):
                if verbose:
                    print(f'weight_gauss_{i}:\t{res.x[imin+2*ngauss+i]}')
                param_names.append(f'weight_gauss_{i}')
            if gmm_prior == 'hierarchical':
                if verbose:
                    print(f'hyper_mu:\t{res.x[imin+3*ngauss-1]}')
                    print(f'hyper_u2:\t{res.x[imin+3*ngauss]}')
                    print(f'hyper_w2:\t{res.x[imin+3*ngauss+1]}')
                param_names.append('hyper_mu')
                param_names.append('hyper_w2')
                param_names.append('hyper_u2')
        
        return res, param_names

    def mcmc(self, params_to_opt, xobs, yobs, errors, nwarm, nsamp, method='mnr',
            ngauss=1, infer_intrinsic=True, num_chains=1, progress_bar=True,
            covmat=False, gmm_prior='hierarchical', seed=1234, verbose=True, init=None,
            include_logdet=True):
        """
        Run an MCMC using the NUTS sampler of ``numpyro`` for the parameters of the
        function given some data, under the assumption of an uncorrelated Gaussian
        likelihood, using the likelihood specififed by 'method'.
        
        Args:
            :params_to_opt (list): The names of the parameters we wish to optimise
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving
                the error on the observed x and y values. Otherwise, this is the
                covariance matrix in the order (x, y)
            :nwarm (int): The number of warmup steps to use in the MCMC
            :nsamp (int): The number of samples to obtain in the MCMC
            :method (str, default='mnr'): The name of the likelihood method to use
                ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more
                information.
            :ngauss (int, default = 1): The number of Gaussians to use in the GMM prior.
                Only used if method='gmm'
            :infer_intrinsic (bool, default=True): Whether to infer the intrinsic
                scatter in the y direction
            :num_chains (int, default=1): The number of independent MCMC chains to run
            :progress_bar (bool, default=True): Whether to display a progress bar for
                the MCMC
            :covmat (bool, default=False): This determines whether the errors argument
                is [xerr, yerr] (False) or a covariance matrix (True).
            :gmm_prior (string, default='hierarchical'): If method='gmm', this decides
                what prior to put on the GMM componenents. If 'uniform', then the mean
                and widths have a uniform prior, and if 'hierarchical' mu and w^2 have a
                Normal and Inverse Gamma prior, respectively.
            :seed (int, default=1234): The seed to use when initialising the sampler
            :verbose (bool, default=True): Whether to print progress or not
            :init (dict, default=None): A dictionary of values of initialise the MCMC at
            :include_logdet (bool, default=True): For the method 'prof', whether to
                include the normalisation term in the likelihood proportional
                to log(det(S))
        Returns:
            :samples (dict): The MCMC samples, where the keys are the parameter names
                and values are ndarrays of the samples
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
            theta = [None] * len(params_to_opt)
            for i, p in enumerate(params_to_opt):
                if ((self.param_prior[p][0] is not None) and
                    (self.param_prior[p][1] is not None)):
                    theta[i] = numpyro.sample(p, dist.Uniform(*self.param_prior[p]))
                else:
                    theta[i] = numpyro.sample(p, dist.ImproperUniform(
                        dist.constraints.real, (), event_shape=()))
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
                if ((self.param_prior['sig'][0] is not None) and
                    (self.param_prior['sig'][1] is not None)):
                    sig = numpyro.sample("sig", dist.Uniform(*self.param_prior['sig']))
                else:
                    sig = numpyro.sample("sig", dist.ImproperUniform(
                        dist.constraints.positive, (), event_shape=()))
            else:
                sig = 0.
                
            # MNR parameters
            if method == 'mnr':
                mu_gauss = numpyro.sample("mu_gauss", dist.ImproperUniform(
                            dist.constraints.real, (), event_shape=()))
                w_gauss = numpyro.sample("w_gauss", dist.ImproperUniform(
                            dist.constraints.positive, (), event_shape=()))
            elif method == 'gmm':
                if gmm_prior == 'uniform':
                    all_mu_gauss = numpyro.sample("mu_gauss", dist.ImproperUniform(
                            dist.constraints.ordered_vector, (), (ngauss,)))
                    all_w_gauss = numpyro.sample("w_gauss", dist.ImproperUniform(
                            dist.constraints.positive, (), (ngauss,)))
                    all_weights = numpyro.sample("weights", dist.Dirichlet(
                            jnp.ones(ngauss)))
                elif gmm_prior == 'hierarchical':
                    hyper_mu = numpyro.sample("hyper_mu", dist.ImproperUniform(
                            dist.constraints.real, (), event_shape=()))
                    hyper_w2 = numpyro.sample("hyper_w2", dist.ImproperUniform(
                            dist.constraints.positive, (), event_shape=()))
                    hyper_u2 = numpyro.sample("hyper_u2", dist.InverseGamma(
                            1/2, hyper_w2/2))
                    all_mu_gauss = numpyro.sample("mu_gauss", roxy.mcmc.OrderedNormal(
                            hyper_mu, jnp.sqrt(hyper_u2)), sample_shape=(ngauss,))
                    all_w_gauss = jnp.sqrt(numpyro.sample("w_gauss", dist.InverseGamma(
                            1/2, hyper_w2/2), sample_shape=(ngauss,)))
                    all_weights = numpyro.sample("weights", dist.Dirichlet(
                            jnp.ones(ngauss)))
                else:
                    raise NotImplementedError

            # Sample
            if method == 'mnr':
                if covmat:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_MNR_MV(xobs, yobs, Sxx, Syy, Sxy, f, G,
                            sig, mu_gauss, w_gauss),
                        obs=yobs,
                    )
                else:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_MNR(xobs, yobs, xerr, yerr, f, fprime,
                            sig, mu_gauss, w_gauss),
                        obs=yobs,
                    )
            elif method == 'unif':
                if covmat:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_unif_MV(xobs, yobs, Sxx, Syy, Sxy, f, G,
                            sig),
                        obs=yobs,
                    )
                else:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_unif(xobs, yobs, xerr, yerr, f, fprime,
                            sig),
                        obs=yobs,
                    )
            elif method == 'prof':
                if covmat:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_prof_MV(xobs, yobs, Sxx, Syy, Sxy, f, G,
                            sig, include_logdet=include_logdet),
                        obs=yobs,
                    )
                else:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_prof(xobs, yobs, xerr, yerr, f, fprime,
                            sig, include_logdet=include_logdet),
                        obs=yobs,
                    )
            elif method == 'gmm':
                if covmat:
                    raise NotImplementedError
                else:
                    numpyro.sample(
                        'obs',
                        roxy.mcmc.Likelihood_GMM(xobs, yobs, xerr, yerr, f, fprime, sig,
                            all_mu_gauss, all_w_gauss, all_weights),
                        obs=yobs,
                    )
            else:
                raise NotImplementedError
             
        rng_key = jax.random.PRNGKey(np.random.randint(seed))
        rng_key, rng_key_ = jax.random.split(rng_key)
        
        try:
            if init is None:
                vals, param_names = self.optimise(params_to_opt, xobs, yobs, errors,
                    method=method, infer_intrinsic=infer_intrinsic, ngauss=ngauss,
                    covmat=covmat, gmm_prior=gmm_prior, verbose=verbose,
                    include_logdet=include_logdet)
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
                if 'sig' in init.keys() and init['sig'] <= 0:
                    warnings.warn('Setting initial sigma to positive value')
                    init['sig'] = 1.e-5
            kernel = numpyro.infer.NUTS(model,
                init_strategy=numpyro.infer.initialization.init_to_value(values=init))
            if verbose:
                print('\nRunning MCMC')
            sampler = numpyro.infer.MCMC(kernel, num_chains=num_chains,
                num_warmup=nwarm, num_samples=nsamp, progress_bar=progress_bar)
            sampler.run(rng_key_)
        except Exception:
            if verbose:
                print('\nCould not init to optimised values')
            kernel = numpyro.infer.NUTS(model)
            if verbose:
                print('\nRunning MCMC')
            sampler = numpyro.infer.MCMC(kernel, num_chains=num_chains,
                num_warmup=nwarm, num_samples=nsamp, progress_bar=progress_bar)
            sampler.run(rng_key_)

        samples = sampler.get_samples()
        
        # We actually samples w2 if gmm_prior = 'hierarchical', so correct for this
        if method == 'gmm' and gmm_prior == 'hierarchical':
            samples['w_gauss'] = jnp.sqrt(samples['w_gauss'])
            
        # Print summary
        if verbose:
            sites = samples
            if isinstance(samples, dict):
                state_sample_field = attrgetter(sampler._sample_field)(
                    sampler._last_state)
                if isinstance(state_sample_field, dict):
                    sites = {
                        k: jnp.expand_dims(v, axis=0)
                        for k, v in samples.items()
                        if k in state_sample_field
                    }
            numpyro.diagnostics.print_summary(sites, prob=0.95)
            extra_fields = sampler.get_extra_fields()
            if ("diverging" in extra_fields):
                print(
                "Number of divergences: {}".format(jnp.sum(extra_fields["diverging"]))
                )
        
        # Raise warning if too few effective samples
        neff = np.zeros(len(samples))
        for i, (k, v) in enumerate(samples.items()):
            x = jnp.expand_dims(v, axis=0)
            try:
                neff[i] = numpyro.diagnostics.effective_sample_size(x)
            except Exception:
                neff[i] = min(numpyro.diagnostics.effective_sample_size(x))
        m = neff < 100
        if m.sum() > 0:
            bad_keys = [k for i,k in enumerate(samples.keys()) if m[i]]
            warnings.warn('Fewer than 100 effective samples for parameters: '
                + ', '.join(bad_keys), category=Warning, stacklevel=2)
            
        # Raise warning if the peak of the posterior is too close to edge of the prior
        bad_keys = []
        for p in params_to_opt:
            if ((self.param_prior[p][0] is not None) and
                (self.param_prior[p][1] is not None)):
                counts, _ = np.histogram(samples[p],
                                np.linspace(self.param_prior[p][0],
                                    self.param_prior[p][1], 30))
                if (np.argmax(counts) < 2) or (np.argmax(counts) > 27):
                    bad_keys.append(p)
        if len(bad_keys) > 0:
            warnings.warn('Posterior near edge of prior for parameters: '
                + ', '.join(bad_keys), category=Warning, stacklevel=2)
            
        # Raise warning if the second derivative of the function is too big
        if covmat:
            xerr = jnp.diag(errors)[:len(xobs)]
        else:
            xerr = errors[0]
            if not hasattr(xerr, 'len'):
                xerr = np.full(len(xobs), xerr)
        # Get medians
        theta = [np.median(samples[p]) for p in params_to_opt]
        t = self.param_default
        t = t.at[pidx].set(theta[:len(pidx)])
        # Check derivatives
        f2prime = self.second_derivative(xobs, t)
        fprime = self.gradient(xobs, t)
        m = (f2prime == 0) & (fprime == 0)  # This case is fine
        if m.sum() < len(xerr):
            crit = np.abs(f2prime[~m] * xerr[~m] / fprime[~m])
            if np.any(crit >= 1):
                nbad = np.sum(crit >= 1)
                warnings.warn(f'Second derivative large for {nbad} data points',
                    category=Warning, stacklevel=2)

        return samples


    def mcmc2opt_index(self, labels, ngauss=1, method='mnr', gmm_prior='hierarchical',
        infer_intrinsic=True):
        """
        Find the indices which convert the samples produced by the MCMC to the order
        required for the optimiser

        Args:
            :labels (list): List of label names in the order produced by the MCMC
            :ngauss (int, default = 1): The number of Gaussians to use in the GMM prior.
                Only used if method='gmm'
            :method (str, default='mnr'): The name of the likelihood method to use
                ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more
                information.
            :gmm_prior (string, default='hierarchical'): If method='gmm', this decides
                what prior to put on the GMM componenents. If 'uniform', then the mean
                and widths have a uniform prior, and if 'hierarchical' mu and w^2 have
                a Normal and Inverse Gamma prior, respectively.
            :infer_intrinsic (bool, default=True): Whether to infer the intrinsic
                scatter in the y direction

        Returns:
            :param_idx (list): The indices which convert the order of parameters from
                the MCMC to that expected by the optimiser
            :param_names (list): The names of the parameters in the order expected by
                the optimiser

        """

        # Now put in order expected by optimisers
        param_idx = [i for i, k in enumerate(labels) if not ((k.startswith('weights')
                or k.startswith('mu_gauss') or k.startswith('w_gauss')
                or k.startswith('sig') or k.startswith('hierarchical')
                or k.startswith('hyper')))]
        if infer_intrinsic:
            param_idx = param_idx + [labels.index('sig')]
        if method == 'gmm':
            param_idx += [labels.index(f'mu_gauss_{i}') for i in range(ngauss)]
            param_idx += [labels.index(f'w_gauss_{i}') for i in range(ngauss)]
            param_idx += [labels.index(f'weights_{i}') for i in range(ngauss-1)]
            if gmm_prior == 'hierarchical':
                param_idx = param_idx + [labels.index('hyper_mu'),
                                        labels.index('hyper_w2'),
                                        labels.index('hyper_u2')]
        elif method == 'mnr':
            param_idx = param_idx + [labels.index('mu_gauss'), labels.index('w_gauss')]
        param_names = [labels[i] for i in param_idx]

        return param_idx, param_names

        
    def compute_information_criterion(self, criterion, params_to_opt, xobs, yobs,
            errors, ngauss=1, infer_intrinsic=True, progress_bar=True, initial=None,
            nwarm=100, nsamp=100, method='mnr', gmm_prior='hierarchical', seed=1234,
            verbose=True, include_logdet=True):
        """
        Compute an information criterion for a given setup
        If an initial guess is not given, we first run a MCMC
        to get an initial guess for the maximum likelihood
        point, and we then an optimsier from this point to get a better
        estimate for this. Since we do not need good MCMC convergence for this,
        small values of nwarm and nsamp can be used.
        
        Args:
            :criterion (str): Which information criterion to use (supported: AIC and
                BIC)
            :params_to_opt (list): The names of the parameters we wish to optimise
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving
                the error on the observed x and y values. Otherwise, this is the
                covariance matrix in the order (x, y)
            :ngauss (int, default = 1): The number of Gaussians to use in the GMM
                prior. Only used if method='gmm'
            :infer_intrinsic (bool, default=True): Whether to infer the intrinsic
                scatter in the y direction
            :initial (jnp.ndarray, default=None): The starting point for the optimised.
                If None, a MCMC is run.
            :progress_bar (bool, default=True): Whether to display a progress bar for
                the MCMC
            :nwarm (int, default=100): The number of warmup steps to use in the MCMC
            :nsamp (int, default=100): The number of samples to obtain in the MCMC
            :method (str, default='mnr'): The name of the likelihood method to use
                ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more
                information.
            :gmm_prior (string, default='hierarchical'): If method='gmm', this decides
                what prior to put on the GMM componenents. If 'uniform', then the mean
                and widths have a uniform prior, and if 'hierarchical' mu and w^2 have
                a Normal and Inverse Gamma prior, respectively.
            :seed (int, default=1234): The seed to use when initialising the sampler
            :verbose (bool, default=True): Whether to print progress or not
            :include_logdet (bool, default=True): For the method 'prof', whether to
                include the normalisation term in the likelihood proportional
                to log(det(S))
            
        Returns:
            :negloglike (float): The optimum negative log-likelihood value
            :metric (float): The value of the information criterion
        
        """
        
        if initial is None:
            # First run a MCMC to get a guess at the peak, catching the low neff warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = self.mcmc(params_to_opt,
                            xobs,
                            yobs,
                            errors,
                            nwarm,
                            nsamp,
                            method=method,
                            ngauss=ngauss,
                            infer_intrinsic=infer_intrinsic,
                            progress_bar=progress_bar,
                            gmm_prior=gmm_prior,
                            seed=seed,
                            verbose=verbose,
                            include_logdet=include_logdet,
                )
            labels, samples = roxy.mcmc.samples_to_array(samples)
            labels = list(labels)
            param_idx, param_names = self.mcmc2opt_index(labels, ngauss=ngauss,
                method=method, gmm_prior=gmm_prior, infer_intrinsic=infer_intrinsic)
            initial = jnp.median(samples[:,param_idx], axis=0)
        
        # Run new optimiser
        res, _ = self.optimise(params_to_opt,
                    xobs,
                    yobs,
                    errors,
                    method=method,
                    infer_intrinsic=infer_intrinsic,
                    initial=initial,
                    ngauss=ngauss,
                    gmm_prior=gmm_prior,
                    verbose=verbose,
                    include_logdet=include_logdet,
        )
        
        # Count number of parameters and get max-likelihood
        npar = len(initial)
        negloglike = res.fun
        
        # Compute criterion
        if criterion == 'AIC':
            metric = 2 * negloglike + 2 * npar
        elif criterion == 'BIC':
            metric = 2 * negloglike + npar * jnp.log(len(xobs))
        else:
            raise NotImplementedError
        
        return negloglike, metric
        
        
    def find_best_gmm(self, params_to_opt, xobs, yobs, xerr, yerr, max_ngauss,
            best_metric='BIC', infer_intrinsic=True, progress_bar=True, nwarm=100,
            nsamp=100, gmm_prior='hierarchical', seed=1234, verbose=True,
            include_logdet=True):
        """
        Find the number of Gaussians to use in a Gaussian Mixture Model
        hyper-prior on the true x values, accoridng to some metric.
        
        Args:
            :params_to_opt (list): The names of the parameters we wish to optimise
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :xerr (jnp.ndarray): The error on the observed x values
            :yerr (jnp.ndarray): The error on the observed y values
            :max_ngauss (int): The maximum number of Gaussians to consider
            :best_metric (str): Metric to use to compare fits (supported: AIC and BIC)
            :infer_intrinsic (bool, default=True): Whether to infer the intrinsic
                scatter in the y direction
            :progress_bar (bool, default=True): Whether to display a progress bar for
                the MCMC
            :nwarm (int, default=100): The number of warmup steps to use in the MCMC
            :nsamp (int, default=100): The number of samples to obtain in the MCMC
            :gmm_prior (string, default='hierarchical'): If method='gmm', this decides
                what prior to put on the GMM componenents. If 'uniform', then the mean
                and widths have a uniform prior, and if 'hierarchical' mu and w^2 have
                a Normal and Inverse Gamma prior, respectively.
            :seed (int, default=1234): The seed to use when initialising the sampler
            :verbose (bool, default=True): Whether to print progress or not
            :include_logdet (bool, default=True): For the method 'prof', whether to
                include the normalisation term in the likelihood proportional
                to log(det(S))
            
        Returns:
            :ngauss (int): The best number of Gaussians to use according to the metric
        
        """
    
        metric = np.empty(max_ngauss)
    
        for ngauss in range(1, max_ngauss+1):
            print('\n' + '*'*20, f'\nStarting ngauss={ngauss}', '\n' + '*'*20 + '\n')
            _, metric[ngauss-1] = self.compute_information_criterion(
                                            best_metric,
                                            params_to_opt,
                                            xobs,
                                            yobs,
                                            [xerr,yerr],
                                            ngauss=ngauss,
                                            infer_intrinsic=infer_intrinsic,
                                            progress_bar=progress_bar,
                                            nwarm=nwarm,
                                            nsamp=nsamp,
                                            method='gmm',
                                            gmm_prior=gmm_prior,
                                            seed=seed,
                                            verbose=verbose,
                                            include_logdet=include_logdet)
        
        ngauss = np.nanargmin(metric) + 1
        if verbose:
            print(f'\nBest ngauss according to {best_metric}:', ngauss)
            metric -= np.amin(metric)
            for i, m in enumerate(metric):
                print(i+1, m)

        return ngauss
