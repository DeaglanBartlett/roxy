import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
from scipy.stats import spearmanr, pearsonr
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
        new_errors[:ny, :ny] = errors[nx:, nx:]
        new_errors[:ny, ny:] = errors[nx:, :nx]
        new_errors[ny:, :ny] = errors[:nx, nx:]
        new_errors[ny:, ny:] = errors[:nx, :nx]
    else:
        new_errors = [errors[1], errors[0]]
    res_xy, names_xy = reg.optimise(param_names, yobs, xobs, new_errors,
                                    method=method, ngauss=ngauss,
                                    covmat=covmat, gmm_prior=gmm_prior)
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
    items = [('y(x) forward', resid_yx_forward, xobs),
             ('y(x) inverse', resid_yx_inverse, xobs),
             ('x(y) forward', resid_xy_forward, yobs),
             ('x(y) inverse', resid_xy_inverse, yobs)]

    results = np.ones(len(items)) * np.inf

    for i, (name, resid, data) in enumerate(items):
        if criterion == 'spearman':
            results[i], pval = spearmanr(data, resid)
            print(
                f"\n{name} Spearman: {round(results[i], 3)}, (p={round(pval, 3)})")
        elif criterion == 'pearson':
            results[i], pval = pearsonr(data, resid)
            print(
                f"\n{name} Pearson: {round(results[i], 3)}, (p={round(pval, 3)})")
        elif criterion == 'hsic':
            stat, results[i] = compute_hsic(np.expand_dims(data, axis=1),
                                            np.expand_dims(resid, axis=1),
                                            alph=0.001)
            if np.isnan(results[i]):
                print(f"\n{name} HSIC: {round(stat, 3)}, (p<0.001)")
                results[i] = 0.
            else:
                print(
                    f"\n{name} HSIC: {round(stat, 3)}, (p={round(results[i], 3)})")
            results[i] = 1 - results[i]
        else:
            raise NotImplementedError

    labels = ['Forward', 'Inverse', 'Forward', 'Inverse']

    if not np.all(np.isnan(results)):
        ibest = np.nanargmin(np.abs(results))
        if ibest in [0, 3]:
            print("\nRecommended direction: y(x)")
        else:
            print("\nRecommended direction: x(y)")
        labels[ibest] += '*'

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    cmap = plt.get_cmap("Set1")

    axs[0, 0].plot(xobs, fun(xobs, theta_yx), label=labels[0], color=cmap(0))
    axs[0, 0].plot(xobs, fun_inv(xobs, theta_xy),
                   label=labels[1], color=cmap(1))
    axs[0, 1].plot(yobs, fun(yobs, theta_xy), label=labels[2], color=cmap(0))
    axs[0, 1].plot(yobs, fun_inv(yobs, theta_yx),
                   label=labels[3], color=cmap(1))
    axs[0, 0].plot(xobs, yobs, '.', color=cmap(2))
    axs[0, 1].plot(yobs, xobs, '.', color=cmap(2))

    axs[1, 0].scatter(xobs, resid_yx_forward, alpha=0.3,
                      label=labels[0], color=cmap(0))
    axs[1, 0].scatter(xobs, resid_yx_inverse, alpha=0.3,
                      label=labels[1], color=cmap(1))
    axs[1, 1].scatter(yobs, resid_xy_forward, alpha=0.3,
                      label=labels[2], color=cmap(0))
    axs[1, 1].scatter(yobs, resid_xy_inverse, alpha=0.3,
                      label=labels[3], color=cmap(1))

    for i in range(axs.shape[1]):
        axs[0, i].sharex(axs[1, i])
        plt.setp(axs[0, i].get_xticklabels(), visible=False)
        axs[1, i].axhline(y=0, color='k')
        axs[0, i].legend()
        axs[1, i].legend()
    axs[0, 0].set_title(r'Infer $y(x)$')
    axs[0, 1].set_title(r'Infer $x(y)$')

    axs[0, 0].set_ylabel(r'$y_{\rm pred}$')
    axs[0, 1].set_ylabel(r'$x_{\rm pred}$')

    axs[1, 0].set_xlabel(r'$x_{\rm obs}$')
    axs[1, 0].set_ylabel(r'Normalised $y$ residuals')
    axs[1, 1].set_xlabel(r'$y_{\rm obs}$')
    axs[1, 1].set_ylabel(r'Normalised $x$ residuals')

    fig.align_labels()
    fig.tight_layout()

    if savename is not None:
        fig.savefig(savename, transparent=False)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())


def compute_hsic(x, y, alph=0.05):
    """
    Python implementation of Hilbert Schmidt Independence Criterion
    using a Gamma approximation. This code is largely taken from
    https://github.com/amber0309/HSIC/tree/master (Shoubo (shoubo.sub AT gmail.com)
    09/11/2016), with the new addition of the computation of the significance level.

    Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B.,
    & Smola, A. J. (2007). A kernel statistical test of independence.
    In Advances in neural information processing systems (pp. 585-592).

    Args:
        :x (np.ndarray): Numpy vector of dependent variable. The row gives the sample
            and the column is the dimension.
        :y (np.ndarray): Numpy vector of independent variable. The row gives the sample
            and the column is the dimension.
        :alph (float): Worst significance level to consider. If the correlation is
            stronger than this, then we don't compute the correlation coefficient.

    Returns:
        :test_stat (float): The test statistic
        :best_alph (float): The significance of this result (if calculated)
    """

    def rbf_dot(pattern1, pattern2, deg):
        size1 = pattern1.shape
        size2 = pattern2.shape
        g = np.sum(pattern1*pattern1, 1).reshape(size1[0], 1)
        h = np.sum(pattern2*pattern2, 1).reshape(size2[0], 1)
        q = np.tile(g, (1, size2[0]))
        r = np.tile(h.T, (size1[0], 1))
        h = q + r - 2 * np.dot(pattern1, pattern2.T)
        h = np.exp(-h/2/(deg**2))
        return h

    n = x.shape[0]

    # width of X
    x_med = x
    g = np.sum(x_med*x_med, 1).reshape(n, 1)
    q = np.tile(g, (1, n))
    r = np.tile(g.T, (n, 1))
    dists = q + r - 2 * np.dot(x_med, x_med.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)
    width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))

    # width of Y
    y_med = y
    g = np.sum(y_med*y_med, 1).reshape(n, 1)
    q = np.tile(g, (1, n))
    r = np.tile(g.T, (n, 1))
    dists = q + r - 2 * np.dot(y_med, y_med.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)
    width_y = np.sqrt(0.5 * np.median(dists[dists > 0]))

    bone = np.ones((n, 1), dtype=float)
    h = np.identity(n) - np.ones((n, n), dtype=float) / n

    k = rbf_dot(x, x, width_x)
    ell = rbf_dot(y, y, width_y)

    kc = np.dot(np.dot(h, k), h)
    lc = np.dot(np.dot(h, ell), h)

    test_stat = np.sum(kc.T * lc) / n

    var_hsic = (kc * lc / 6)**2

    var_hsic = (np.sum(var_hsic) - np.trace(var_hsic)) / n / (n-1)

    var_hsic = var_hsic * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

    k = k - np.diag(np.diag(k))
    ell = ell - np.diag(np.diag(ell))

    mu_x = np.dot(np.dot(bone.T, k), bone) / n / (n-1)
    mu_y = np.dot(np.dot(bone.T, ell), bone) / n / (n-1)

    m_hsic = (1 + mu_x * mu_y - mu_x - mu_y) / n

    al = m_hsic**2 / var_hsic
    bet = var_hsic*n / m_hsic

    thresh = scipy.stats.gamma.ppf(1-alph, al, scale=bet)[0][0]

    # Find threshold of significance for sufficiently weak correlation
    if test_stat < thresh:
        def to_zero(a):
            r = scipy.stats.gamma.ppf(a, al, scale=bet)[0][0] - test_stat
            return r
        res = scipy.optimize.root_scalar(to_zero, x0=1-alph)
        best_alph = 1 - res.root
    else:
        best_alph = np.nan

    return test_stat, best_alph
