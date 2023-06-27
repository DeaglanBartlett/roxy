from jax import lax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.util import promote_shapes
import numpy as np

import roxy.likelihoods

class Likelihood_MNR(dist.Distribution):
   
    def __init__(self, xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss):
        self.xobs, self.yobs, self.xerr, self.yerr, self.f, self.fprime, self.sig, self.mu_gauss, self.w_gauss = promote_shapes(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs),
            jnp.shape(yobs),
            jnp.shape(xerr),
            jnp.shape(yerr),
            jnp.shape(f),
            jnp.shape(fprime),
            jnp.shape(sig),
            jnp.shape(mu_gauss),
            jnp.shape(w_gauss),
        )
        super(Likelihood_MNR, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_mnr(self.xobs, self.yobs, self.xerr, self.yerr, self.f, self.fprime, self.sig, self.mu_gauss, self.w_gauss)  # ADD ARGS HERE
        
        
class Likelihood_profile(dist.Distribution):
   
    def __init__(self, xobs, yobs, xerr, yerr, f, fprime, sig):
        self.xobs, self.yobs, self.xerr, self.yerr, self.f, self.fprime, self.sig = promote_shapes(xobs, yobs, xerr, yerr, f, fprime, sig)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs),
            jnp.shape(yobs),
            jnp.shape(xerr),
            jnp.shape(yerr),
            jnp.shape(f),
            jnp.shape(fprime),
            jnp.shape(sig),
        )
        super(Likelihood_profile, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_profile(self.xobs, self.yobs, self.xerr, self.yerr, self.f, self.fprime, self.sig)
        

class Likelihood_uniform(dist.Distribution):
   
    def __init__(self, xobs, yobs, xerr, yerr, f, fprime, sig):
        self.xobs, self.yobs, self.xerr, self.yerr, self.f, self.fprime, self.sig = promote_shapes(xobs, yobs, xerr, yerr, f, fprime, sig)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs),
            jnp.shape(yobs),
            jnp.shape(xerr),
            jnp.shape(yerr),
            jnp.shape(f),
            jnp.shape(fprime),
            jnp.shape(sig),
        )
        super(Likelihood_uniform, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_uniform(self.xobs, self.yobs, self.xerr, self.yerr, self.f, self.fprime, self.sig)
        
        
def samples_to_array(samples):

    keys = list(samples.keys())

    # Get labels and length of vector for each parameter
    labels = []
    nparam = np.zeros(len(keys), dtype=int)
        
    for m in range(len(keys)):
        if len(samples[keys[m]].shape) == 1:
            labels += [keys[m]]
            nparam[m] = 1
        else:
            nparam[m] = samples[keys[m]].shape[1]
            labels += [keys[m] + '_%i'%n for n in range(nparam[m])]
            
    nparam = [0] + list(np.cumsum(nparam))

    all_samples = np.empty((samples[keys[0]].shape[0], len(labels)))        # Flatten the samples array so it is (# samples, # parameters)
    for m in range(len(keys)):
        if len(samples[keys[m]].shape) == 1:
            all_samples[:,nparam[m]] = samples[keys[m]][:]
        else:
            for n in range(nparam[m+1]-nparam[m]):
                all_samples[:,nparam[m]+n] = samples[keys[m]][:,n]
                
    labels = np.array(labels)
    all_samples = np.array(all_samples)

    return labels, all_samples
