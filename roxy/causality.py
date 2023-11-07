import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
from scipy.stats import spearmanr,pearsonr
from roxy.regressor import RoxyRegressor

def assess_causality(fun, fun_inv, xobs, yobs, errors, param_names, param_default,
    param_prior, method='mnr', criterion='hsic', ngauss=1, covmat=False,
    gmm_prior='hierarchical', savename=None, show=True):
    """
    Due to the asymmetry between x and y, this function assesses whether one
    should fit y(x) or x(y) with the intrinsic scatter in the dependent variable.
    The code runs both forward fits, i.e. y(x) and x(y), and the reverse fits,
    where one uses the parameters obtained from the forward fit and uses the
    inverse function. The correlation coefficient of the residuals
    (normalised by the square root of the sum of the squares of the vertical
    errors and the intrindic scatter) is then found, and the method which minimsies this
    recommended as the best option. Plots of the fits and the residuals
    are produced, since these can be more informative for some functions. The
    recommended setup is indicated by a star in the plots.
    
    Args:
        :fun (callable): The function, f, to be considered, y = f(x, theta).
            The function must take two arguments, the first of which
            is the independent variable, the second of which are the parameters (as an
            array or list).
        :fun_inv (callable): If `fun` is y = f(x, theta), then this callable gives
            x = f^{-1}(y, theta) for the same theta. The function must take two
            arguments, the first corresponding to y and the second is theta.
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :errors (jnp.ndarray): If covmat=False, then this is [xerr, yerr], giving
            the error on the observed x and y values. Otherwise, this is the
            covariance matrix in the order (x, y)
        :param_names (list): The list of parameter names, in the order which they are
            supplied to fun
        :param_default (list): The default valus of the parameters
        :param_prior (dict): The prior range for each of the parameters. The prior is
            assumed to be uniform in this range. If either entry is None in the prior,
            then an infinite uniform prior is assumed.
        :method (str, default='mnr'): The name of the likelihood method to use
            ('mnr', 'gmm', 'unif' or 'prof'). See ``roxy.likelihoods`` for more
            information
        :criterion (str, default='hsic'): The method used to determine the weakest
            correlation of the residuals. One of 'hsic', 'spearman' or 'pearson'.
        :ngauss (int, default = 1): The number of Gaussians to use in the GMM prior.
            Only used if method='gmm'
        :covmat (bool, default=False): This determines whether the errors argument
            is [xerr, yerr] (False) or a covariance matrix (True).
        :gmm_prior (string, default='hierarchical'): If method='gmm', this decides
            what prior to put on the GMM componenents. If 'uniform', then the mean
            and widths have a uniform prior, and if 'hierarchical' mu and w^2 have
            a Normal and Inverse Gamma prior, respectively.
        :savename (str, default=None): If not None, save the figure to the file given by
            this argument.
        :show (bool, default=True): If True, display the figure with plt.show()
    """
    
    reg = RoxyRegressor(fun, param_names, param_default, param_prior)
    
    # y vs x
    print('\nFitting y vs x')
    res_yx, names_yx = reg.optimise(param_names, xobs, yobs, errors, method=method,
                        ngauss=ngauss, covmat=covmat, gmm_prior=gmm_prior)
    theta_yx = [None] * len(param_names)
    for i, t in enumerate(param_names):
        j = names_yx.index(t)
        theta_yx[j] = res_yx.x[i]
    theta_yx = np.array(theta_yx)
    sig_yx = res_yx.x[names_yx.index('sig')]
    
    # x vs y
    print('\nFitting x vs y')
    if covmat:
        new_errors = np.empty(errors.shape)
        nx = len(xobs)
        ny = len(yobs)
        new_errors[:ny,:ny] = errors[nx:,nx:]
        new_errors[:ny,ny:] = errors[nx:,:nx]
        new_errors[ny:,:ny] = errors[:nx,nx:]
        new_errors[ny:,ny:] = errors[:nx,:nx]
    else:
        new_errors = [errors[1], errors[0]]
    res_xy, names_xy = reg.optimise(param_names, yobs, xobs, new_errors,
                    method=method, ngauss=ngauss, covmat=covmat, gmm_prior=gmm_prior)
    theta_xy = [None] * len(param_names)
    for i, t in enumerate(param_names):
        j = names_xy.index(t)
        theta_xy[j] = res_xy.x[i]
    theta_xy = np.array(theta_xy)
    sig_xy = res_xy.x[names_xy.index('sig')]
    
    # Get normalisation for residuals
    if covmat:
        nx = len(xobs)
        xscale = np.sqrt(np.diag(errors)[:nx] + sig_xy ** 2)
        yscale = np.sqrt(np.diag(errors)[nx:] + sig_yx ** 2)
    else:
        xscale = np.sqrt(errors[0] ** 2 + sig_xy ** 2)
        yscale = np.sqrt(errors[1] ** 2 + sig_yx ** 2)
    
    # Residuals
    resid_yx_forward = (yobs - fun(xobs, theta_yx)) / yscale
    resid_yx_inverse = (yobs - fun_inv(xobs, theta_xy)) / yscale
    resid_xy_forward = (xobs - fun(yobs, theta_xy)) / xscale
    resid_xy_inverse = (xobs - fun_inv(yobs, theta_yx)) / xscale
    items = [('y vs x forward', resid_yx_forward, xobs),
             ('y vs x inverse', resid_yx_inverse, xobs),
             ('x vs y forward', resid_xy_forward, yobs),
             ('x vs y inverse', resid_xy_inverse, yobs)]
    
    results = np.ones(len(items)) * np.inf
    
    for i, (name, resid, data) in enumerate(items):
        if criterion == 'spearman':
            results[i] = spearmanr(data, resid)[0]
            print(f"\n{name} Spearman:", round(results[i],3))
        elif criterion == 'pearson':
            results[i] = pearsonr(data, resid)[0]
            print(f"{name} Pearson:", round(results[i],3))
        elif criterion == 'hsic':
            stat, results[i] = compute_hsic(np.expand_dims(data, axis=1),
                        np.expand_dims(resid, axis=1),
                        alph=0.05)
            print(f"{name} HSIC: {round(stat,3)}, (conf={round(results[i],3)})")
            results[i] = 1 - results[i]
        else:
            raise NotImplementedError
        
    
    labels = ['Forward', 'Inverse', 'Forward', 'Inverse']
    
    if not np.all(np.isnan(results)):
        ibest = np.nanargmin(np.abs(results))
        print("\nRecommended direction:", items[ibest][0])
        labels[ibest] += '*'
        
    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10,6))
        
    cmap = plt.get_cmap("Set1")
    
    
    axs[0,0].plot(xobs, fun(xobs, theta_yx), label=labels[0], color=cmap(0))
    axs[0,0].plot(xobs, fun_inv(xobs, theta_xy), label=labels[1], color=cmap(1))
    axs[0,1].plot(yobs, fun(yobs, theta_xy), label=labels[2], color=cmap(0))
    axs[0,1].plot(yobs, fun_inv(yobs, theta_yx), label=labels[3], color=cmap(1))
    axs[0,0].plot(xobs, yobs, '.', color=cmap(2))
    axs[0,1].plot(yobs, xobs, '.', color=cmap(2))
    
    axs[1,0].scatter(xobs, resid_yx_forward, alpha=0.3, label=labels[0], color=cmap(0))
    axs[1,0].scatter(xobs, resid_yx_inverse, alpha=0.3, label=labels[1], color=cmap(1))
    axs[1,1].scatter(yobs, resid_xy_forward, alpha=0.3, label=labels[2], color=cmap(0))
    axs[1,1].scatter(yobs, resid_xy_inverse, alpha=0.3, label=labels[3], color=cmap(1))
    
    for i in range(axs.shape[1]):
        axs[0,i].sharex(axs[1,i])
        plt.setp(axs[0,i].get_xticklabels(), visible=False)
        axs[1,i].axhline(y=0, color='k')
        axs[0,i].legend()
        axs[1,i].legend()
    axs[0,0].set_title(r'Infer $y(x)$')
    axs[0,1].set_title(r'Infer $x(y)$')
        
    axs[0,0].set_ylabel(r'$y_{\rm pred}$')
    axs[0,1].set_ylabel(r'$x_{\rm pred}$')
        
    axs[1,0].set_xlabel(r'$x_{\rm obs}$')
    axs[1,0].set_ylabel(r'Normalised $y$ residuals')
    axs[1,1].set_xlabel(r'$y_{\rm obs}$')
    axs[1,1].set_ylabel(r'Normalised $x$ residuals')
        
    fig.align_labels()
    fig.tight_layout()
    
    if savename is not None:
        fig.savefig(savename, transparent=False)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())
    
    return


def compute_hsic(X, Y, alph = 0.05):
    """
    Python implementation of Hilbert Schmidt Independence Criterion
    using a Gamma approximation. This code is largely taken from
    https://github.com/amber0309/HSIC/tree/master (Shoubo (shoubo.sub AT gmail.com)
    09/11/2016), with the new addition of the computation of the significance level.

    Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B.,
    & Smola, A. J. (2007). A kernel statistical test of independence.
    In Advances in neural information processing systems (pp. 585-592).
    
    Args:
        :X (np.ndarray): Numpy vector of dependent variable. The row gives the sample
            and the column is the dimension.
        :Y (np.ndarry): Numpy vector of independent variable. The row gives the sample
            and the column is the dimension.
        :alph (float): Worst significance level to consider. If the correlation is
            stronger than this, then we don't compute the correlation coefficient.
    
    Returns:
        :testStat (float): The test statistic
        :best_alph (float): The significance of this result (if calculated)
    """
    
    def rbf_dot(pattern1, pattern2, deg):
        size1 = pattern1.shape
        size2 = pattern2.shape
        G = np.sum(pattern1*pattern1, 1).reshape(size1[0],1)
        H = np.sum(pattern2*pattern2, 1).reshape(size2[0],1)
        Q = np.tile(G, (1, size2[0]))
        R = np.tile(H.T, (size1[0], 1))
        H = Q + R - 2* np.dot(pattern1, pattern2.T)
        H = np.exp(-H/2/(deg**2))
        return H

    n = X.shape[0]

    # width of X
    Xmed = X
    G = np.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )
    dists = Q + R - 2* np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)
    width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )


    # width of Y
    Ymed = Y
    G = np.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )
    dists = Q + R - 2* np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)
    width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )

    bone = np.ones((n, 1), dtype = float)
    H = np.identity(n) - np.ones((n,n), dtype = float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6)**2

    varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

    varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC**2 / varHSIC
    bet = varHSIC*n / mHSIC

    thresh = scipy.stats.gamma.ppf(1-alph, al, scale=bet)[0][0]
    
    # Find threshold of significance for sufficiently weak correlation
    if testStat < thresh:
        def to_zero(a):
            r = scipy.stats.gamma.ppf(a, al, scale=bet)[0][0] - testStat
            return r
        res = scipy.optimize.root_scalar(to_zero, x0=1-alph)
        best_alph = 1 - res.root
    else:
        best_alph = np.nan

    return testStat, best_alph
