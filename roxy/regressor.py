import jax
from scipy.optimize import minimize
import numpy as np

import roxy.likelihoods

class RoxyRegressor():

    def __init__(self, fun, param_names, param_default, param_prior):
        
        self.single_fun = fun
        self.single_gradfun = jax.grad(self.single_fun, argnums=0)
        self.fun = jax.vmap(self.single_fun, (0, None), 0)
        self.gradfun = jax.vmap(self.single_gradfun, (0, None), 0)
        
        self.param_names = param_names
        self.param_default = np.array(param_default)
        self.param_prior = param_prior
        
    def value(self, x, theta):
        return self.fun(x, theta)
        
    def gradient(self, x, theta):
        return self.gradfun(x, theta)
        
    def negloglike(self, theta, xobs, yobs, xerr, yerr, sig=0., mu_gauss=0., w_gauss=1., method='mnr'):
        f = self.value(xobs, theta)
        fprime = self.gradient(xobs, theta)
        
        if method == 'uniform':
            return roxy.likelihoods.negloglike_uniform(xobs, yobs, xerr, yerr, f, fprime, sig)
        elif method == 'profile':
            return roxy.likelihoods.negloglike_profile(xobs, yobs, xerr, yerr, f, fprime, sig)
        elif method == 'mnr':
            return roxy.likelihoods.negloglike_mnr(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss)
        else:
            raise NotImplementedError
            
    def optimise(self, params_to_opt, xobs, yobs, xerr, yerr, method='mnr', infer_intrinsic=True, initial=None):
        """
        scipy.optimize._optimize.OptimizeResult
        """
    
        # Get indices of params to optimise
        pidx = [self.param_names.index(p) for p in params_to_opt]
        if len(pidx) != len(self.param_names):
            print('Not optimising all parameters. Using defaults:')
            for pname, pdefault in zip(self.param_names, self.param_default):
                if pname not in params_to_opt:
                    print(f'{pname}:\t{pdefault}')
        
        def fopt(theta):
            
            # Parameters of function
            t = self.param_default
            t[pidx] = theta[:len(pidx)]
            
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
                           
            # f(x) and f'(x) for these params
            f = self.value(xobs, t)
            fprime = self.gradient(xobs, t)
            
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
        print('\nResults:')
        for p, val in zip(params_to_opt, res.x):
            print(f'{p}:\t{val}')
        if infer_intrinsic:
            print(f'sig:\t{res.x[len(params_to_opt)]}')
        if method == 'mnr':
            print(f'mu_gauss:\t{res.x[-2]}')
            print(f'w_gauss:\t{res.x[-1]}')
        
        return res
