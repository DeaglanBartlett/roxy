import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import numpy as np
import numpyro
import numpyro.distributions as dist

import roxy.likelihoods
import roxy.mcmc

class RoxyRegressor():

    def __init__(self, fun, param_names, param_default, param_prior):
        
        self.single_fun = fun
        self.single_gradfun = jax.grad(self.single_fun, argnums=0)
        self.fun = jax.vmap(self.single_fun, (0, None), 0)
        self.gradfun = jax.vmap(self.single_gradfun, (0, None), 0)
        
        self.param_names = param_names
        self.param_default = jnp.array(param_default)
        self.param_prior = param_prior
        
    def value(self, x, theta):
        return self.fun(x, theta)
        
    def gradient(self, x, theta):
        return self.gradfun(x, theta)
        
    def negloglike(self, theta, xobs, yobs, xerr, yerr, sig=0., mu_gauss=0., w_gauss=1., method='mnr'):
        f = self.value(xobs, theta)
        fprime = self.gradient(xobs, theta)
        
        if sig < 0. or (method == 'mnr' and w_gauss < 0.):
            return np.nan
        
        if method == 'mnr':
            return roxy.likelihoods.negloglike_mnr(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss)
        elif method == 'uniform':
            return roxy.likelihoods.negloglike_uniform(xobs, yobs, xerr, yerr, f, fprime, sig)
        elif method == 'profile':
            return roxy.likelihoods.negloglike_profile(xobs, yobs, xerr, yerr, f, fprime, sig)
        else:
            raise NotImplementedError
            
    def get_param_index(self, params_to_opt, verbose=True):
        # Get indices of params to optimise
        pidx = [self.param_names.index(p) for p in params_to_opt if p in self.param_names]
        if len(pidx) != len(self.param_names) and verbose:
            print('\nNot optimising all parameters. Using defaults:')
            for pname, pdefault in zip(self.param_names, self.param_default):
                if pname not in params_to_opt:
                    print(f'{pname}:\t{pdefault}')
        return jnp.array(pidx)
            
    def optimise(self, params_to_opt, xobs, yobs, xerr, yerr, method='mnr', infer_intrinsic=True, initial=None):
        """
        scipy.optimize._optimize.OptimizeResult
        """
    
        # Get indices of params to optimise
        pidx = self.get_param_index(params_to_opt)
        
        def fopt(theta):
            
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
            
            # MNR parameters
            if method == 'mnr':
                mu_gauss = theta[-2]
                w_gauss = theta[-1]
            else:
                mu_gauss, w_gauss = None, None
            
            return self.negloglike(t, xobs, yobs, xerr, yerr, sig=sig, mu_gauss=mu_gauss, w_gauss=w_gauss, method=method)
        
        # Get initial guess
        if initial is None:
            initial = [np.random.uniform(*self.param_prior[p]) for p in params_to_opt]
            if infer_intrinsic:
                initial = initial + [np.random.uniform(*self.param_prior['sig'])]
            if method == 'mnr':
                initial = initial + [np.mean(xobs), np.std(xobs)]
            
        res = minimize(fopt, initial, method="Nelder-Mead")
        
        # Print results
        print('\nOptimisation Results:')
        for p, val in zip(params_to_opt, res.x):
            print(f'{p}:\t{val}')
        if infer_intrinsic:
            print(f'sig:\t{res.x[len(params_to_opt)]}')
        if method == 'mnr':
            print(f'mu_gauss:\t{res.x[-2]}')
            print(f'w_gauss:\t{res.x[-1]}')
        
        return res

    def mcmc(self, params_to_opt, xobs, yobs, xerr, yerr, nwarm, nsamp, method='mnr', infer_intrinsic=True, progress_bar=True):

        pidx = self.get_param_index(params_to_opt, verbose=False)
                
        def model():
        
            # Parameters of function
            theta = [numpyro.sample(p, dist.Uniform(*self.param_prior[p])) for p in params_to_opt]
            t = self.param_default
            t = t.at[pidx].set(theta[:len(pidx)])
            
            # f(x) and f'(x) for these params
            f = self.value(xobs, t)
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

            # Sample
            if method == 'mnr':
                numpyro.sample(
                    'obs',
                    roxy.mcmc.Likelihood_MNR(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss),
                    obs=yobs,
                )
            elif method == 'uniform':
                numpyro.sample(
                    'obs',
                    roxy.mcmc.Likelihood_uniform(xobs, yobs, xerr, yerr, f, fprime, sig),
                    obs=yobs,
                )
            elif method == 'profile':
                numpyro.sample(
                    'obs',
                    roxy.mcmc.Likelihood_profile(xobs, yobs, xerr, yerr, f, fprime, sig),
                    obs=yobs,
                )
            else:
                raise NotImplementedError
             
        rng_key = jax.random.PRNGKey(np.random.randint(1234))
        rng_key, rng_key_ = jax.random.split(rng_key)
        
        try:
            vals = self.optimise(params_to_opt, xobs, yobs, xerr, yerr, method=method, infer_intrinsic=infer_intrinsic).x
            kernel = numpyro.infer.NUTS(model, init_strategy=numpyro.infer.initialization.init_to_value(values=vals))
            print('\nRunning MCMC')
            sampler = numpyro.infer.MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, progress_bar=progress_bar)
            sampler.run(rng_key_)
        except:
            print('\nCould not init to optimised values')
            kernel = numpyro.infer.NUTS(model)
            print('\nRunning MCMC')
            sampler = numpyro.infer.MCMC(kernel, num_warmup=nwarm, num_samples=nsamp, progress_bar=progress_bar)
            sampler.run(rng_key_)

        sampler.print_summary()
        samples = sampler.get_samples()

        return samples
