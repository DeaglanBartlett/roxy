from jax import lax
import jax.random
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.util import promote_shapes
import numpy as np
from numpyro.distributions.util import validate_sample
import scipy.optimize
from numpyro.distributions.util import is_prng_key
from jax.scipy.stats import norm as jax_norm
from jax.scipy.special import ndtri, ndtr

import roxy.likelihoods

class Likelihood_MNR(dist.Distribution):
    """
    Class to be used by ``numpyro`` to evaluate the log-likelihood under
    the assumption of an uncorrelated Gaussian likelihood with a Gaussian
    prior on the true x positions.
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated
            at xobs
        :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx
            evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        :mu_gauss (float): The mean of the Gaussian prior on the true x positions
        :w_gauss (float): The standard deviation of the Gaussian prior on the true x
            positions
    """
   
    def __init__(self, xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss):
        self.xobs, self.yobs, self.xerr, self.yerr, self.f, self.fprime, self.sig, \
            self.mu_gauss, self.w_gauss = promote_shapes(xobs, yobs, xerr, yerr, f,
            fprime, sig, mu_gauss, w_gauss)
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
        return - roxy.likelihoods.negloglike_mnr(self.xobs, self.yobs, self.xerr,
                self.yerr, self.f, self.fprime, self.sig, self.mu_gauss, self.w_gauss)
        
        
class Likelihood_MNR_MV(dist.Distribution):
    """
    Class to be used by ``numpyro`` to evaluate the log-likelihood under
    the assumption of a correlated Gaussian likelihood (i.e. arbitrary
    covariance matrix), with a Gaussian prior on the true x positions.
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :Sxx (jnp.ndarray): The xx component of the covariance matrix giving the errors
            on the observed (x, y) values
        :Syy (jnp.ndarray): The yy component of the covariance matrix giving the errors
            on the observed (x, y) values
        :Sxy (jnp.ndarray): The xy component of the covariance matrix giving the errors
            on the observed (x, y) values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at
            xobs
        :G (jnp.ndarray): If we are fitting the function f(x), this is
            G_{ij} = df_i/dx_j evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        :mu_gauss (float): The mean of the Gaussian prior on the true x positions
        :w_gauss (float): The standard deviation of the Gaussian prior on the true x
            positions
    """
        
    def __init__(self, xobs, yobs, Sxx, Syy, Sxy, f, G, sig, mu_gauss, w_gauss):
        xobs_p = xobs[..., jnp.newaxis]
        yobs_p = yobs[..., jnp.newaxis]
        f_p = f[..., jnp.newaxis]
        xobs_p, yobs_p, Sxx, Syy, Sxy, f_p, self.G, \
            sig, mu_gauss, w_gauss = promote_shapes(xobs_p, yobs_p, Sxx, Syy, Sxy,
            f_p, G, sig, mu_gauss, w_gauss)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs_p)[:-2],
            jnp.shape(yobs_p)[:-2],
            jnp.shape(Sxx)[:-2],
            jnp.shape(Syy)[:-2],
            jnp.shape(Sxy)[:-2],
            jnp.shape(f_p)[:-2],
            jnp.shape(self.G)[:-2],
            jnp.shape(sig)[:-2],
            jnp.shape(mu_gauss)[:-2],
            jnp.shape(w_gauss)[:-2],
            )
        event_shape = jnp.shape(xobs_p)[-1:]
        self.xobs = xobs_p[...,0]
        self.yobs = yobs_p[...,0]
        self.Sigma = jnp.concatenate(
                        [jnp.concatenate([Sxx, Sxy], axis=-1),
                        jnp.concatenate([Sxy.T, Syy], axis=-1)]
                    )
        self.f = f_p[...,0]
        self.sig = sig[...,0]
        self.mu_gauss = mu_gauss[...,0]
        self.w_gauss = w_gauss[...,0]
        super(Likelihood_MNR_MV, self).__init__(
            batch_shape = batch_shape,
            event_shape = event_shape
        )
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_mnr_mv(self.xobs, self.yobs, self.Sigma,
            self.f, self.G, self.sig, self.mu_gauss, self.w_gauss)
        
        
class Likelihood_prof(dist.Distribution):
    """
    Class to be used by ``numpyro`` to evaluate the log-likelihood under
    the assumption of an uncorrelated Gaussian likelihood, evaluated at
    the maximum likelihood values of xtrue (the profile likelihood)

    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated
            at xobs
        :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx
            evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        :include_logdet (bool, default=True): Whether to include the normalisation term
            in the likelihood proportional to log(det(S))
    """
        
    def __init__(self, xobs, yobs, xerr, yerr, f, fprime, sig, include_logdet=True):
        self.xobs, self.yobs, self.xerr, self.yerr, \
        self.f, self.fprime, self.sig = promote_shapes(xobs, yobs, xerr, yerr,
        f, fprime, sig)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs),
            jnp.shape(yobs),
            jnp.shape(xerr),
            jnp.shape(yerr),
            jnp.shape(f),
            jnp.shape(fprime),
            jnp.shape(sig),
        )
        self.include_logdet = include_logdet
        super(Likelihood_prof, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_prof(self.xobs, self.yobs, self.xerr,
            self.yerr, self.f, self.fprime, self.sig,
            include_logdet=self.include_logdet)
        
        
class Likelihood_prof_MV(dist.Distribution):
    """
    Class to be used by ``numpyro`` to evaluate the log-likelihood under
    the assumption of a correlated Gaussian likelihood (i.e. arbitrary covariance
    matrix), evaluated at the maximum likelihood values of xtrue (the profile
    likelihood)
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :Sxx (jnp.ndarray): The xx component of the covariance matrix giving the errors
            on the observed (x, y) values
        :Syy (jnp.ndarray): The yy component of the covariance matrix giving the errors
            on the observed (x, y) values
        :Sxy (jnp.ndarray): The xy component of the covariance matrix giving the errors
            on the observed (x, y) values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at
            xobs
        :G (jnp.ndarray): If we are fitting the function f(x), this is
            G_{ij} = df_i/dx_j evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        :include_logdet (bool, default=True): Whether to include the normalisation term
            in the likelihood proportional to log(det(S))
    """
   
    def __init__(self, xobs, yobs, Sxx, Syy, Sxy, f, G, sig,
        include_logdet = True):
        xobs_p = xobs[..., jnp.newaxis]
        yobs_p = yobs[..., jnp.newaxis]
        f_p = f[..., jnp.newaxis]
        xobs_p, yobs_p, Sxx, Syy, Sxy, f_p, self.G, sig = promote_shapes(xobs_p, yobs_p,
            Sxx, Syy, Sxy, f_p, G, sig)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs_p)[:-2],
            jnp.shape(yobs_p)[:-2],
            jnp.shape(Sxx)[:-2],
            jnp.shape(Syy)[:-2],
            jnp.shape(Sxy)[:-2],
            jnp.shape(f_p)[:-2],
            jnp.shape(self.G)[:-2],
            jnp.shape(sig)[:-2],
            )
        event_shape = jnp.shape(xobs_p)[-1:]
        self.xobs = xobs_p[...,0]
        self.yobs = yobs_p[...,0]
        self.Sigma = jnp.concatenate(
                        [jnp.concatenate([Sxx, Sxy], axis=-1),
                        jnp.concatenate([Sxy.T, Syy], axis=-1)]
                    )
        self.f = f_p[...,0]
        self.sig = sig[...,0]
        self.include_logdet = include_logdet
        super(Likelihood_prof_MV, self).__init__(
            batch_shape = batch_shape,
            event_shape = event_shape
        )
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_prof_mv(self.xobs, self.yobs, self.Sigma,
            self.f, self.G, self.sig, include_logdet=self.include_logdet)
    
        

class Likelihood_unif(dist.Distribution):
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
        :f (jnp.ndarray): If we are fitting the funciton f(x), this is f(x) evaluated
            at xobs
        :fprime (jnp.ndarray): If we are fitting the funciton f(x), this is df/dx
            evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
    """
        
    def __init__(self, xobs, yobs, xerr, yerr, f, fprime, sig):
        self.xobs, self.yobs, self.xerr, self.yerr, \
            self.f, self.fprime, self.sig = promote_shapes(xobs, yobs, xerr, yerr,
            f, fprime, sig)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs),
            jnp.shape(yobs),
            jnp.shape(xerr),
            jnp.shape(yerr),
            jnp.shape(f),
            jnp.shape(fprime),
            jnp.shape(sig),
        )
        super(Likelihood_unif, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_unif(self.xobs, self.yobs, self.xerr,
            self.yerr, self.f, self.fprime, self.sig)
        
        
class Likelihood_unif_MV(dist.Distribution):
    """
    Class to be used by ``numpyro`` to evaluate the log-likelihood under
    the assumption of a correlated Gaussian likelihood (i.e. arbitrary covariance
    matrix), where we have marginalised over the true x values, assuming an
    infinite uniform prior on these.
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :Sxx (jnp.ndarray): The xx component of the covariance matrix giving the errors
            on the observed (x, y) values
        :Syy (jnp.ndarray): The yy component of the covariance matrix giving the errors
            on the observed (x, y) values
        :Sxy (jnp.ndarray): The xy component of the covariance matrix giving the errors
            on the observed (x, y) values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at
            xobs
        :G (jnp.ndarray): If we are fitting the function f(x), this is
            G_{ij} = df_i/dx_j evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
    """
        
    def __init__(self, xobs, yobs, Sxx, Syy, Sxy, f, G, sig):
        xobs_p = xobs[..., jnp.newaxis]
        yobs_p = yobs[..., jnp.newaxis]
        f_p = f[..., jnp.newaxis]
        xobs_p, yobs_p, Sxx, Syy, Sxy, f_p, self.G, sig = promote_shapes(xobs_p, yobs_p,
            Sxx, Syy, Sxy, f_p, G, sig)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs_p)[:-2],
            jnp.shape(yobs_p)[:-2],
            jnp.shape(Sxx)[:-2],
            jnp.shape(Syy)[:-2],
            jnp.shape(Sxy)[:-2],
            jnp.shape(f_p)[:-2],
            jnp.shape(self.G)[:-2],
            jnp.shape(sig)[:-2],
            )
        event_shape = jnp.shape(xobs_p)[-1:]
        self.xobs = xobs_p[...,0]
        self.yobs = yobs_p[...,0]
        self.Sigma = jnp.concatenate(
                        [jnp.concatenate([Sxx, Sxy], axis=-1),
                        jnp.concatenate([Sxy.T, Syy], axis=-1)]
                    )
        self.f = f_p[...,0]
        self.sig = sig[...,0]
        super(Likelihood_unif_MV, self).__init__(
            batch_shape = batch_shape,
            event_shape = event_shape
        )
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_unif_mv(self.xobs, self.yobs, self.Sigma,
            self.f, self.G, self.sig)
        
        
class Likelihood_GMM(dist.Distribution):
    """
    Class to be used by ``numpyro`` to evaluate the log-likelihood under
    the assumption of an uncorrelated Gaussian likelihood with a Gaussian
    Mixture Model prior on the true x positions.

    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated
            at xobs
        :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx
            evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        :all_mu_gauss (jnp.ndarray): The mean of the Gaussians in the GMM prior on the
            true x positions
        :all_w_gauss (jnp.ndarray): The standard deviation of the Gaussians in the GMM
            prior on the true x positions
        :all_weights (jnp.ndarray): The weights of the Gaussians in the GMM prior on the
            true x positions
    """
        
    def __init__(self, xobs, yobs, xerr, yerr, f, fprime, sig, all_mu_gauss,
        all_w_gauss, all_weights):

        self.xobs, self.yobs, self.xerr, self.yerr, self.f, self.fprime, self.sig = \
            promote_shapes(xobs, yobs, xerr, yerr, f, fprime, sig)
        
        self.all_mu_gauss, self.all_w_gauss, self.all_weights = promote_shapes(
            all_mu_gauss, all_w_gauss, all_weights)
        
        batch_shape = lax.broadcast_shapes(
            jnp.shape(xobs),
            jnp.shape(yobs),
            jnp.shape(xerr),
            jnp.shape(yerr),
            jnp.shape(f),
            jnp.shape(fprime),
            jnp.shape(sig),
            (),
            (),
            ()
        )
        super(Likelihood_GMM, self).__init__(batch_shape = batch_shape,)

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        return - roxy.likelihoods.negloglike_gmm(self.xobs, self.yobs, self.xerr,
            self.yerr, self.f, self.fprime, self.sig, self.all_mu_gauss,
            self.all_w_gauss, self.all_weights)


def samples_to_array(samples):
    """
    Converts a dictionary of samples returned by ``numpro`` to a list of names
    and an array of samples.
    
    Args:
        :samples (dict): The MCMC samples, where the keys are the parameter names and
            values are ndarrays of the samples
        
    Returns:
        :labels (np.ndarray): The names of the sampled variables
        :all_samples (np.ndarray): The sampled values for these variables.
            Shape = (number of samples, number of parameters).
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

    all_samples = np.empty((samples[keys[0]].shape[0], len(labels)))
    # Flatten the samples array so it is (# samples, # parameters)
    for m in range(len(keys)):
        if len(samples[keys[m]].shape) == 1:
            all_samples[:,nparam[m]] = samples[keys[m]][:]
        else:
            for n in range(nparam[m+1]-nparam[m]):
                all_samples[:,nparam[m]+n] = samples[keys[m]][:,n]
                
    labels = np.array(labels)
    all_samples = np.array(all_samples)

    return labels, all_samples
    
    
def compute_bias(samples, truths, verbose=True):
    """
    Computes the bias between MCMC samples and the true values of the parameters.
    For the intrinsic scatter, a truncated normal distribution is first fitted to
    the parameters since this parameter can only take positive values.
    
    Args:
        :samples (dict): The MCMC samples, where the keys are the parameter names and
            values are ndarrays of the samples
        :truths (dict): The true values of the parameters. The keys should be a subset
            of the keys of samples
        :verbose (bool, default=True): Whether to print biases or not
            
    Reurns:
        : biases (dict): The biases in each of the parameters (units=number of sigmas)
    """
    
    biases = {}
    
    for k, v in truths.items():
        
        if k == 'sig':
        
            # Fit these samples to a truncated Gaussian
            def negloglike(pars):
                mu, sig = pars
                nll = (
                    np.log(2) - 0.5 * np.log(2 * np.pi * sig ** 2)
                    - np.log(1 + scipy.special.erf(mu / np.sqrt(2) / sig))
                    - (samples[k] - mu) ** 2 / 2 / sig ** 2
                )
                return - np.sum(nll)
            initial = [np.mean(samples[k]), np.std(samples[k])]
            bounds = [(None, None), (0, None)]  # sigma must be >= 0
            res = scipy.optimize.minimize(negloglike, initial, bounds=bounds,
                method='nelder-mead')
            mu, sig = res.x
            if verbose:
                print('Truncated normal fit for sig:', mu, sig)

        else:
            mu = float(np.mean(samples[k]))
            sig = float(np.std(samples[k]))
            
        biases[k] = (mu - v) / sig
    
    if verbose:
        print('\nComputed biases (units=sigma):')
        for k, b in biases.items():
            print(f'{k}:\t{b}')

    return biases
    
    
class OrderedNormal(dist.Distribution):
    arg_constraints = {"loc": dist.constraints.real, "scale": dist.constraints.positive}
    support = dist.constraints.ordered_vector
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc=0.0, scale=1.0, *, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(OrderedNormal, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        eps = jax.random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        res = self.loc + eps * self.scale
        return jnp.sort(res)

    @validate_sample
    def log_prob(self, value):
        normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * self.scale)
        value_scaled = (value - self.loc) / self.scale
        return -0.5 * value_scaled**2 - normalize_term

    def cdf(self, value):
        scaled = (value - self.loc) / self.scale
        return ndtr(scaled)

    def log_cdf(self, value):
        return jax_norm.logcdf(value, loc=self.loc, scale=self.scale)

    def icdf(self, q):
        return self.loc + self.scale * ndtri(q)

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(self.scale**2, self.batch_shape)
