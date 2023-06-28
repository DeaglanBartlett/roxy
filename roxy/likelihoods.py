import jax.numpy as jnp

def negloglike_mnr(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss):
    """
    Computes the negative log-likelihood under the assumption of an uncorrelated
    Gaussian likelihood with a Gaussian prior on the true x positions.
    
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
        
    Returns:
        :ngeglogP (float): The negative log-likelihood
    """
    N = len(xobs)
    Ai = fprime
    if len(Ai) == 1:
        Ai = jnp.full(N, Ai[0])
    Bi = f - Ai * xobs
    
    s2 = yerr ** 2 + sig ** 2
    den = Ai ** 2 * w_gauss ** 2 * xerr ** 2 + s2 * (w_gauss ** 2 + xerr ** 2)
    
    neglogP = (
        N / 2 * jnp.log(2 * jnp.pi)
        + 1/2 * jnp.sum(jnp.log(den))
        + 1/2 * jnp.sum(w_gauss ** 2 * (Ai * xobs + Bi - yobs) ** 2 / den)
        + 1/2 * jnp.sum(xerr ** 2 * (Ai * mu_gauss + Bi - yobs) ** 2 / den)
        + 1/2 * jnp.sum(s2 * (xobs - mu_gauss) ** 2 / den)
    )

    return neglogP
    
    
def negloglike_profile(xobs, yobs, xerr, yerr, f, fprime, sig):
    """
    Computes the negative log-likelihood under the assumption of an uncorrelated
    Gaussian likelihood, evaluated at the maximum likelihood values of xtrue
    (the profile likelihood)
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at xobs
        :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        
    Returns:
        :ngeglogP (float): The negative log-likelihood
    """
    N = len(xobs)
    Ai = fprime
    Bi = f - Ai * xobs
    sigy = jnp.atleast_1d(jnp.sqrt(yerr ** 2 + sig ** 2))
    if len(sigy) == 1:
        sigy = jnp.full(N, sigy[0])
    
    neglogP = (
        N / 2 * jnp.log(2 * jnp.pi)
        + jnp.sum(jnp.log(sigy))
        + 1/2 * jnp.sum((Ai * xobs + Bi - yobs) ** 2 / (Ai ** 2 * xerr ** 2 + sigy ** 2))
    )

    return neglogP


def negloglike_uniform(xobs, yobs, xerr, yerr, f, fprime, sig):
    """
    Computes the negative log-likelihood under the assumption of an uncorrelated
    Gaussian likelihood, where we have marginalised over the true x values,
    assuming an infinite uniform prior on these.
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at xobs
        :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        
    Returns:
        :ngeglogP (float): The negative log-likelihood
    """
    N = len(xobs)
    Ai = jnp.atleast_1d(fprime)
    if len(Ai) == 1:
        Ai = jnp.full(N, Ai[0])
    Bi = f - Ai * xobs

    neglogP = (
        N / 2 * jnp.log(2 * jnp.pi)
        + 1/2 * jnp.sum(jnp.log(Ai ** 2 * xerr ** 2 + yerr ** 2 + sig ** 2))
        + 1/2 * jnp.sum((Ai * xobs + Bi - yobs) ** 2 / (Ai ** 2 * xerr ** 2 + yerr ** 2 + sig ** 2))
    )

    return neglogP
