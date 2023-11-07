import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
from roxy.regressor import RoxyRegressor

def assess_causality(fun, fun_inv, xobs, yobs, errors, param_names, param_default,
    param_prior, method='mnr', ngauss=1, covmat=False, gmm_prior='hierarchical',
    savename=None, show=True):
    """
    Due to the asymmetry between x and y, this function assesses whether one
    should fit y(x) or x(y) with the intrinsic scatter in the dependent variable.
    The code runs both forward fits, i.e. y(x) and x(y), and the reverse fits,
    where one uses the parameters obtained from the forward fit and uses the
    inverse function. The Spearman and Pearson correlation coefficients of the
    residuals is then found and the method which minimsies the means of these
    is recommended as the best option. Plots of the fits and the residuals are
    produced, since these can be more informative for some functions.
    
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
    
    # x vs y
    print('\nFitting x vs y')
    if not covmat:
        new_errors = [errors[1], errors[0]]
    else:
        new_errors = np.empty(errors.shape)
        nx = len(xobs)
        ny = len(yobs)
        new_errors[:ny,:ny] = errors[nx:,nx:]
        new_errors[:ny,ny:] = errors[nx:,:nx]
        new_errors[ny:,:ny] = errors[:nx,nx:]
        new_errors[ny:,ny:] = errors[:nx,:nx]
    res_xy, names_xy = reg.optimise(param_names, yobs, xobs, new_errors,
                    method=method, ngauss=ngauss, covmat=covmat, gmm_prior=gmm_prior)
    theta_xy = [None] * len(param_names)
    for i, t in enumerate(param_names):
        j = names_xy.index(t)
        theta_xy[j] = res_xy.x[i]
    theta_xy = np.array(theta_xy)
    
    # Residuals
    resid_yx_forward = yobs - fun(xobs, theta_yx)
    resid_yx_inverse = yobs - fun_inv(xobs, theta_xy)
    resid_xy_forward = xobs - fun(yobs, theta_xy)
    resid_xy_inverse = xobs - fun_inv(yobs, theta_yx)
    items = [('y vs x forward', resid_yx_forward, xobs),
             ('y vs x inverse', resid_yx_inverse, xobs),
             ('x vs y forward', resid_xy_forward, yobs),
             ('x vs y inverse', resid_xy_inverse, yobs)]
    
    results = np.ones(len(items)) * np.inf
    
    for i, (name, resid, data) in enumerate(items):
        spear = spearmanr(data, resid)[0]
        pears = pearsonr(data, resid)[0]
        results[i] = (spear + pears) / 2
        print(f"\n{name} Spearman:", round(spear,3))
        print(f"{name} Pearson:", round(pears,3))
        
    ibest = np.nanargmin(np.abs(results))
    print("\nRecommended direction:", items[ibest][0])
    
    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10,6))
        
    cmap = plt.get_cmap("Set1")
    
    labels = ['Forward', 'Inverse', 'Forward', 'Inverse']
    labels[ibest] += '*'
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
    axs[1,0].set_ylabel(r'$y$ Residuals')
    axs[1,1].set_xlabel(r'$y_{\rm obs}$')
    axs[1,1].set_ylabel(r'$x$ Residuals')
        
    fig.align_labels()
    fig.tight_layout()
    
    if savename is not None:
        fig.savefig(savename, transparent=False)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())
    
    return
