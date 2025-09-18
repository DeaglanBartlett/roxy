import jax.numpy as jnp
import warnings

def likelihood_warnings(method, infer_intrinsic, nx, errors, covmat):
    """
    Raise warnings if using asymptotically biased likelihood
    for the situation of interest.
    
    Args:
        :method (str, default='mnr'): The name of the likelihood method to use
                ('mnr', 'gmm', 'unif' or 'prof').
        :infer_intrinsic (bool, default=True): Whether to infer the intrinsic
                scatter in the y direction
        :nx (int): The number of observed x values
        :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving
                the error on the observed x and y values. Otherwise, this is the
                covariance matrix in the order (x, y)
        :covmat (bool, default=False): This determines whether the errors argument
                is [xerr, yerr] (False) or a covariance matrix (True).
    """
    
    # Modify the warnings filter to always show all warnings
    warnings.filterwarnings('always')
    
    # Check if xerrs are all zero
    if covmat:
        xerr = errors[:nx,:nx]
    else:
        xerr = errors[0]
    if isinstance(xerr, (float, int)):
        no_xerr = (xerr == 0)
    else:
        no_xerr = jnp.all(jnp.array(xerr) == 0)
        
    warning_message = None
    
    if no_xerr and method not in ['unif', 'prof']:
        warning_message = (
            f'Not recommended method "{method}" for this setup. '
            'Use "unif" or "prof" instead.')
    elif (not no_xerr) and infer_intrinsic and method not in ['mnr', 'gmm']:
        warning_message = (
            f'Not recommended method "{method}" for this setup. '
            'Use "mnr" or "gmm" instead.')
    elif (not no_xerr) and (not infer_intrinsic) and method != 'prof':
        warning_message = (
            f'Not recommended method "{method}" for this setup. '
            'Use "prof" instead.')
    
    if warning_message is not None:
        warnings.warn(warning_message, UserWarning)

    return

def negloglike_mnr(xobs, yobs, xerr, yerr, f, fprime, sig, mu_gauss, w_gauss):
    """
    Computes the negative log-likelihood under the assumption of an uncorrelated
    Gaussian likelihood with a Gaussian prior on the true x positions.
    
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
        
    Returns:
        :neglogP (float): The negative log-likelihood
    """
    N = len(xobs)
    Ai = fprime
    if (not hasattr(Ai, "__len__")) or len(Ai) == 1:
        Ai = jnp.full(N, jnp.squeeze(jnp.array(Ai)))
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
    
    
def negloglike_gmm(xobs, yobs, xerr, yerr, f, fprime, sig, all_mu_gauss, all_w_gauss,
    all_weights):
    """
    Computes the negative log-likelihood under the assumption of an uncorrelated
    Gaussian likelihood with a GMM prior on the true x positions.
    
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
        :all_mu_gauss (jnp.ndarray): The means of the Gaussians in the GMM prior on
            the true x positions
        :all_w_gauss (jnp.ndarray): The standard deviations of the Gaussians in the GMM
            prior on the true x positions
        :all_weights (jnp.ndarray): The weights of the Gaussians in the GMM prior on the
            true x positions
        
    Returns:
        :neglogP (float): The negative log-likelihood
    """
    
    ngauss = len(all_weights)
    N = len(xobs)
    all_logP = jnp.empty((ngauss, N))
    
    for i in range(ngauss):
    
        mu_gauss = all_mu_gauss[i]
        w_gauss = all_w_gauss[i]
        weight = all_weights[i]
    
        Ai = fprime
        if (not hasattr(Ai, "__len__")) or len(Ai) == 1:
            Ai = jnp.full(N, jnp.squeeze(jnp.array(Ai)))
        Bi = f - Ai * xobs
        
        s2 = yerr ** 2 + sig ** 2
        den = Ai ** 2 * w_gauss ** 2 * xerr ** 2 + s2 * (w_gauss ** 2 + xerr ** 2)
        
        all_logP = all_logP.at[i,:].set(
            - jnp.log(weight)
            + 1/2 * jnp.log(2 * jnp.pi)
            + 1/2 * jnp.log(den)
            + 1/2 * (w_gauss ** 2 * (Ai * xobs + Bi - yobs) ** 2 / den)
            + 1/2 * (xerr ** 2 * (Ai * mu_gauss + Bi - yobs) ** 2 / den)
            + 1/2 * (s2 * (xobs - mu_gauss) ** 2 / den)
        )
                                    
    all_logP = - all_logP
    
    # Combine the Gaussians
    max_logP = jnp.amax(all_logP, axis=0)
    neglogP = - (max_logP + jnp.log(jnp.sum(jnp.exp(all_logP - max_logP), axis=0)))
    neglogP = jnp.sum(neglogP)
    
    return neglogP
    
    
def negloglike_prof(xobs, yobs, xerr, yerr, f, fprime, sig, include_logdet=True):
    """
    Computes the negative log-likelihood under the assumption of an uncorrelated
    Gaussian likelihood, evaluated at the maximum likelihood values of xtrue
    (the profile likelihood)
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at
            xobs
        :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx
            evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        :include_logdet (bool, default=True): Whether to include the normalisation term
            in the likelihood proportional to log(det(S))
        
    Returns:
        :neglogP (float): The negative log-likelihood
    """
    N = len(xobs)
    Ai = fprime
    Bi = f - Ai * xobs
    sigy = jnp.atleast_1d(jnp.sqrt(yerr ** 2 + sig ** 2))
    if len(sigy) == 1:
        sigy = jnp.full(N, sigy[0])
    
    if include_logdet:
        neglogP = (
            N / 2 * jnp.log(2 * jnp.pi)
            + jnp.sum(jnp.log(sigy))
            + 1/2 * jnp.sum((Ai * xobs + Bi - yobs) ** 2 /
            (Ai ** 2 * xerr ** 2 + sigy ** 2))
        )
    else:
        neglogP = (
            1/2 * jnp.sum((Ai * xobs + Bi - yobs) ** 2 /
            (Ai ** 2 * xerr ** 2 + sigy ** 2))
        )

    return neglogP


def negloglike_unif(xobs, yobs, xerr, yerr, f, fprime, sig):
    """
    Computes the negative log-likelihood under the assumption of an uncorrelated
    Gaussian likelihood, where we have marginalised over the true x values,
    assuming an infinite uniform prior on these.
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at
            xobs
        :fprime (jnp.ndarray): If we are fitting the function f(x), this is df/dx
            evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        
    Returns:
        :neglogP (float): The negative log-likelihood
    """
    N = len(xobs)
    Ai = jnp.atleast_1d(fprime)
    if (not hasattr(Ai, "__len__")) or len(Ai) == 1:
        Ai = jnp.full(N, jnp.squeeze(jnp.array(Ai)))
    Bi = f - Ai * xobs

    neglogP = (
        N / 2 * jnp.log(2 * jnp.pi)
        + 1/2 * jnp.sum(jnp.log(Ai ** 2 * xerr ** 2 + yerr ** 2 + sig ** 2))
        + 1/2 * jnp.sum((Ai * xobs + Bi - yobs) ** 2 /
        (Ai ** 2 * xerr ** 2 + yerr ** 2 + sig ** 2))
    )

    return neglogP


def check_valid_covmat(D, tol=1e-8):
    """
    Check if a covariance matrix is valid (symmetric and positive semi-definite)
    - Symmetry (within tolerance)
    - Positive semi-definiteness (eigvals >= -tol)
    
    Args:
        :D (jnp.ndarray): The covariance matrix to check
        :tol (float, default=1e-8): The tolerance for numerical checks

    Returns:
        :is_valid (bool): Whether the covariance matrix is valid
    """

    # Symmetry check
    symmetric = jnp.allclose(D, D.T, atol=tol)

    # PSD check
    eigvals = jnp.linalg.eigvalsh(D)
    psd = jnp.all(eigvals >= -tol)

    return jnp.logical_and(symmetric, psd)
    
    
def negloglike_mnr_mv(xobs, yobs, Sigma, f, G, sig, mu_gauss, w_gauss):
    """
    Computes the negative log-likelihood under the assumption of a correlated
    Gaussian likelihood (i.e. arbitrary covariance matrix) with a Gaussian prior
    on the true x positions.
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :Sigma (jnp.ndarray): The covariance matrix giving the errors on the observed
            (x, y) values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at
            xobs
        :G (jnp.ndarray): If we are fitting the function f(x), this is
            G_{ij} = df_i/dx_j evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        :mu_gauss (float): The mean of the Gaussian prior on the true x positions
        :w_gauss (float): The standard deviation of the Gaussian prior on the true x
            positions
        
    Returns:
        :neglogP (float): The negative log-likelihood
    """
    
    nx = len(xobs)
    ny = len(yobs)
    W = jnp.identity(nx) * w_gauss ** 2
    GW = jnp.matmul(G, W)
    
    # Covariance
    M = Sigma + jnp.concatenate([
                            jnp.concatenate([W, GW.T], axis=-1),
                            jnp.concatenate([GW, jnp.matmul(GW, G.T)
                            + jnp.identity(ny) * sig ** 2], axis=-1)
                            ])
    _, logdet2piM = jnp.linalg.slogdet(2 * jnp.pi * M)
    Minv = jnp.linalg.inv(M)
    
    # Vector
    z = jnp.concatenate([mu_gauss - xobs, f + jnp.matmul(G, mu_gauss - xobs) - yobs])
    
    neglogP = 1/2 * logdet2piM + 1/2 * jnp.sum(z * jnp.matmul(Minv, z))

    # Penalise invalid covariance matrices
    is_valid = check_valid_covmat(M)
    penalty = 1e20
    
    return jnp.where(is_valid, neglogP, penalty)

    
    
def negloglike_prof_mv(xobs, yobs, Sigma, f, G, sig, include_logdet=True):
    """
    Computes the negative log-likelihood under the assumption of a correlated
    Gaussian likelihood (i.e. arbitrary covariance matrix), evaluated at the
    maximum likelihood values of xtrue (the profile likelihood)
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :Sigma (jnp.ndarray): The covariance matrix giving the errors on the observed
            (x, y) values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated at
            xobs
        :G (jnp.ndarray): If we are fitting the function f(x), this is
            G_{ij} = df_i/dx_j evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        :include_logdet (bool, default=True): Whether to include the normalisation term
            in the likelihood proportional to log(det(S))
        
    Returns:
        :neglogP (float): The negative log-likelihood
    """
    
    nx = len(xobs)
    ny = len(yobs)
    D = (
        Sigma[nx:,nx:] + + jnp.identity(ny) * sig ** 2
        + jnp.matmul(G, jnp.matmul(Sigma[:nx,:nx], G.T))
        - jnp.matmul(Sigma[nx:,:nx], G.T) - jnp.matmul(G, Sigma[:nx,nx:])
    )
    S = jnp.array(Sigma)
    S = S.at[nx:,nx:].set(S[nx:,nx:] + jnp.identity(ny) * sig ** 2)
    _, logdet2piS = jnp.linalg.slogdet(2 * jnp.pi * S)
    
    Dinv = jnp.linalg.inv(D)

    z = f - yobs
    if include_logdet:
        neglogP = 1/2 * logdet2piS + 1/2 * jnp.sum(z * jnp.matmul(Dinv, z))
    else:
        neglogP = 1/2 * jnp.sum(z * jnp.matmul(Dinv, z))

    # Penalise invalid covariance matrices
    is_valid = check_valid_covmat(D)
    penalty = 1e20
    
    return jnp.where(is_valid, neglogP, penalty)

    
def negloglike_unif_mv(xobs, yobs, Sigma, f, G, sig):
    """
    Computes the negative log-likelihood under the assumption of a correlated
    Gaussian likelihood (i.e. arbitrary covariance matrix), where we have
    marginalised over the true x values, assuming an infinite uniform prior on these.
    
    Args:
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :Sigma (jnp.ndarray): The covariance matrix giving the errors on the observed
            (x, y) values
        :f (jnp.ndarray): If we are fitting the function f(x), this is f(x) evaluated
            at xobs
        :G (jnp.ndarray): If we are fitting the function f(x), this is
            G_{ij} = df_i/dx_j evaluated at xobs
        :sig (float): The intrinsic scatter, which is added in quadrature with yerr
        
    Returns:
        :neglogP (float): The negative log-likelihood
    """
    
    nx = len(xobs)
    ny = len(yobs)
    D = (
        Sigma[nx:,nx:] + + jnp.identity(ny) * sig ** 2
        + jnp.matmul(G, jnp.matmul(Sigma[:nx,:nx], G.T))
        - jnp.matmul(Sigma[nx:,:nx], G.T) - jnp.matmul(G, Sigma[:nx,nx:])
    )
    _, logdet2piD = jnp.linalg.slogdet(2 * jnp.pi * D)

    Dinv = jnp.linalg.inv(D)
    
    z = f - yobs
    neglogP = 1/2 * logdet2piD + 1/2 * jnp.sum(z * jnp.matmul(Dinv, z))
        
    # Penalise invalid covariance matrices
    is_valid = check_valid_covmat(D)
    penalty = 1e20
    
    return jnp.where(is_valid, neglogP, penalty)


