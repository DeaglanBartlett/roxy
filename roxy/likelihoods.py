import jax.numpy as jnp

def negloglike_mnr(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss):
    """
    Return - log(like)
    """
    if sig < 0:
        return jnp.nan
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
    Return - log(like)
    """
    if sig < 0:
        return jnp.nan
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
    Return - log(like)
    """
    if sig < 0:
        return jnp.nan
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
