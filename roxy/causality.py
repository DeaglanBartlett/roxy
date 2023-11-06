import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr

from roxy.regressor import RoxyRegressor


def assess_causality(fun, fun_inv, xobs, yobs, errors, param_names, theta0, param_prior, method='mnr', ngauss=1, covmat=False, gmm_prior='hierarchical'):
    
    reg = RoxyRegressor(fun, param_names, theta0, param_prior)
    
    # y vs x
    print('\nFitting y vs x')
    res_yx, names_yx = reg.optimise(param_names, xobs, yobs, errors, method=method, ngauss=ngauss, covmat=covmat, gmm_prior=gmm_prior)
    theta_yx = [None] * len(param_names)
    for i, t in enumerate(param_names):
        j = names_yx.index(t)
        theta_yx[j] = res_yx.x[i]
    theta_yx = np.array(theta_yx)
    
    # x vs y
    print('\nFitting x vs y')
    if not covmat:
        res_xy, names_xy = reg.optimise(param_names, yobs, xobs, [errors[1], errors[0]], method=method, ngauss=ngauss, covmat=covmat, gmm_prior=gmm_prior)
    else:
        raise NoteImplementedError
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
        
    i = np.nanargmin(np.abs(results))
    print("\nRecommended direction:", items[i][0])
    
    # Plot
    fig, axs = plt.subplots(3, 2, figsize=(8,8))
    
    axs[0,0].plot(xobs, fun(xobs, theta_yx), label='Forward')
    axs[0,0].plot(xobs, fun_inv(xobs, theta_xy), label='Inverse')
    axs[0,1].plot(yobs, fun(yobs, theta_xy), label='Forward')
    axs[0,1].plot(yobs, fun_inv(yobs, theta_yx), label='Inverse')
    axs[0,0].plot(xobs, yobs, '.')
    axs[0,1].plot(yobs, xobs, '.')
    
    axs[1,0].scatter(xobs, resid_yx_forward, alpha=0.3, label="Forward")
    axs[1,0].scatter(xobs, resid_yx_inverse, alpha=0.3, label="Inverse")
    axs[1,1].scatter(xobs, resid_xy_forward, alpha=0.3, label="Forward")
    axs[1,1].scatter(xobs, resid_xy_inverse, alpha=0.3, label="Inverse")
    
    axs[2,0].scatter(yobs, resid_yx_forward, alpha=0.3, label="Forward")
    axs[2,0].scatter(yobs, resid_yx_inverse, alpha=0.3, label="Inverse")
    axs[2,1].scatter(yobs, resid_xy_forward, alpha=0.3, label="Forward")
    axs[2,1].scatter(yobs, resid_xy_inverse, alpha=0.3, label="Inverse")
    
    for ax in axs.flatten():
        ax.axhline(y=0, color='k')
        ax.legend()
    for i in range(axs.shape[0]):
        axs[i,0].set_title(r'Infer $y(x)$')
        axs[i,1].set_title(r'Infer $x(y)$')
        
    axs[0,0].set_xlabel(r'$x_{\rm obs}$')
    axs[0,1].set_xlabel(r'$y_{\rm obs}$')
    axs[0,0].set_ylabel(r'$y_{\rm pred}$')
    axs[0,1].set_ylabel(r'$x_{\rm pred}$')
        
    axs[1,0].set_xlabel(r'$x_{\rm obs}$')
    axs[1,1].set_xlabel(r'$x_{\rm obs}$')
    axs[1,0].set_ylabel(r'$x$ Residuals')
    axs[1,1].set_ylabel(r'$y$ Residuals') 
    
    axs[2,0].set_xlabel(r'$y_{\rm obs}$')
    axs[2,1].set_xlabel(r'$y_{\rm obs}$') 
    axs[2,0].set_ylabel(r'$x$ Residuals')
    axs[2,1].set_ylabel(r'$y$ Residuals') 
        
    fig.tight_layout()
    plt.show()
    
    # REMOVE THE DIAGONAL PLOTS
    
    return