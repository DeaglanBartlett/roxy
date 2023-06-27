import jax
import roxy.likelihoods

class RoxyRegressor():

    def __init__(self, fun):
        
        self.single_fun = fun
        self.single_gradfun = jax.grad(self.single_fun, argnums=0)
        self.fun = jax.vmap(self.single_fun, (0, None), 0)
        self.gradfun = jax.vmap(self.single_gradfun, (0, None), 0)
        
    def value(self, x, theta):
        return self.fun(x, theta)
        
    def gradient(self, x, theta):
        return self.gradfun(x, theta)
        
    def negloglike(self, theta, xobs, yobs, xerr, yerr, sig=0., mu_gauss=0., w_gauss=1., method='uniform'):
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
