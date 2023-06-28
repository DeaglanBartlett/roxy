from jax import lax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.util import promote_shapes
import numpy as np

import roxy.likelihoods

class Likelihood_MNR(dist.Distribution):
   
    def __init__(self, xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss):
        """
        Class to be used by ``numpyro`` to evaluate the log-likelihood under
        the assumption of an uncorrelated Gaussian likelihood with a Gaussian
        prior on the true x positions.
        
        Args:
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :xerr (jnp.ndarray): The error on the observed x values
            :yerr (jnp.ndarray): The error on the observed y values
            :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at xobs
            :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx evaluated at xobs
            :sig (float): The intrinsic scatter, which is added in quadrature with yerr
            :mu_gauss (float): The mean of the Gaussian prior on the true x positions
            :w_gauss (float): The standard deviation of the Gaussian prior on the true x positions
        """
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
        """
        Class to be used by ``numpyro`` to evaluate the log-likelihood under
        the assumption of an uncorrelated Gaussian likelihood, evaluated at
        the maximum likelihood values of xtrue (the profile likelihood)
        
        Args:
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :xerr (jnp.ndarray): The error on the observed x values
            :yerr (jnp.ndarray): The error on the observed y values
            :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at xobs
            :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx evaluated at xobs
            :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        """
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
        """
        Class to be used by ``numpyro`` to evaluate the log-likelihood under
        the assumption of an uncorrelated Gaussian likelihood, where we have
        marginalised over the true x values, assuming an infinite uniform
        prior on these.
        
        Args:
            :xobs (jnp.ndarray): The observed x values
            :yobs (jnp.ndarray): The observed y values
            :xerr (jnp.ndarray): The error on the observed x values
            :yerr (jnp.ndarray): The error on the observed y values
            :f (jnp.ndarray): If we are fitting the funciton f(x), this is f(x) evaluated at xobs
            :fprime (jnp.ndarray): If we are fitting the funciton f(x), this is df/dx evaluated at xobs
            :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        """
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
    """
    Converts a dictionary of samples returned by ``numpro`` to a list of names
    and an array of samples.
    
    Args:
        :samples (dict): The MCMC samples, where the keys are the parameter names and values are ndarrays of the samples
        
    Returns:
        :labels (np.ndarray): The names of the sampled variables
        :all_samples (np.ndarray): The sampled values for these variables. Shape = (number of samples, number of parameters).
    """

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
