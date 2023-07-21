import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import roxy.mcmc
from getdist import plots, MCSamples
import arviz as az
from fgivenx import plot_contours

rcParams['text.usetex'] = True
rcParams.update({'font.size': 14})

def triangle_plot(samples, labels=None, to_plot='all', module='corner', param_prior=None, savename=None, show=True):
    """
    Plot the 1D and 2D posterior distributions of the parameters in a triangle plot.
    
    Args:
        :samples (dict): The MCMC samples, where the keys are the parameter names and values are ndarrays of the samples
        :labels (list, default=None): List of parameter labels ot use in the plot. If None, then use the names given as keys in samples.
        :to_plot (list, default='all'): If 'all', then use all parameters. If a list, then only use the parameters given in that list
        :module (str, default='corner'): Which module to use to make the triangle plot ('corner' or 'getdist' currently available)
        :param_prior (dict, default=None): If not None and using 'getdist', use this to specify the range of the varibales to prevent undesirable smoothing effects.
        :savename (str, default=None): If not None, save the figure to the file given by this argument.
        :show (bool, default=True): If True, display the figure with plt.show()
    """
    names, all_samples = roxy.mcmc.samples_to_array(samples)
    
    if to_plot != 'all':
        idx = [np.squeeze(np.where(names==p)) for p in to_plot]
        names = names[idx]
        all_samples = all_samples[:,idx]
    
    if labels is None:
        labs = list(names)
        for p, l in zip(['mu_gauss', 'w_gauss', 'sig'], [r'\mu_{\rm gauss}', r'w_{\rm gauss}', r'\sigma_{\rm int}']):
            if (p in names) and ((p in to_plot) or (to_plot == 'all')):
                i = np.squeeze(np.where(names==p))
                labs[i] = l
                
        # GMM parameters
        if 'weights_0' in names:
            # Extract number of Gaussians
            ngauss = len([n for n in names if n.startswith('weights')])
            for i in range(ngauss):
                for p, l in zip([f'mu_gauss_{i}', f'w_gauss_{i}', f'weights_{i}'], [r'\mu_{%i}'%i, r'w_{%i}'%i, r'\nu_{%i}'%i]):
                    j = np.squeeze(np.where(names==p))
                    labs[j] = l
        
        # Kelly prior parameters
        if 'hyper_mu' in names:
            for p, l in zip(['hyper_mu', 'hyper_w2', 'hyper_u2'], [r'\mu_\star', r'w_\star^2', r'u_\star^2']):
                j = np.squeeze(np.where(names==p))
                labs[j] = l
        
    else:
        labs = [labels[n] for n in names]
        
    if module == 'corner':
        labs = ['$'+l+'$' for l in labs]
        fig, _ = plt.subplots(len(labs), len(labs), figsize=(8,8))
        corner.corner(all_samples, labels=labs, fig=fig)
    elif module == 'getdist':
    
        if param_prior is None:
            ranges = {}
        else:
            ranges = param_prior
        ranges['w_gauss'] = [0, None]
        
        if 'weights_0' in names:
            for i in range(ngauss):
                ranges[f'w_gauss_{i}'] = [0, None]
                ranges[f'weights_{i}'] = [0, 1]
                
        if 'hyper_mu' in names:
            ranges['hyper_w2'] = [0, None]
            ranges['hyper_u2'] = [0, None]
    
        samps = MCSamples(
            samples=all_samples,
            names=names,
            labels=labs,
            ranges=ranges
        )
            
        g = plots.get_subplot_plotter(width_inch=8)
        g.triangle_plot(samps, filled=True)
            
    else:
        raise NotImplementedError
    plt.gcf().align_labels()
    
    if savename is not None:
        plt.savefig(savename, transparent=True)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())
    
    return
    
    
def trace_plot(samples, labels=None, to_plot='all', savename=None, show=True):
    """
    Plot the trace of the parameter values as a function of MCMC step
    
    Args:
        :samples (dict): The MCMC samples, where the keys are the parameter names and values are ndarrays of the samples
        :labels (list, default=None): List of parameter labels ot use in the plot. If None, then use the names given as keys in samples.
        :to_plot (list, default='all'): If 'all', then use all parameters. If a list, then only use the parameters given in that list
        :savename (str, default=None): If not None, save the figure to the file given by this argument.
        :show (bool, default=True): If True, display the figure with plt.show()
    """

    # Check for GMM
    if 'weights' in samples.keys():
        new_samples = samples.copy()
        for k in ['mu_gauss', 'w_gauss', 'weights']:
            new_samples.pop(k)
            v = samples[k]
            for i in range(v.shape[1]):
                new_samples[f'{k}_{i}'] = v[:,i]
        npar = len(new_samples.keys())
        res = az.from_dict(new_samples)
    else:
        res = az.from_dict(samples)
        npar = len(samples.keys())
    
    if to_plot != 'all':
        npar = len(to_plot)
    figsize = (12, min(2 * npar, 10))
    
    if to_plot == 'all':
        az.plot_trace(res, compact=True, figsize=figsize)
    else:
        az.plot_trace(res, compact=True, var_names=to_plot, figsize=figsize)
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, transparent=True)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())

    return
    
def posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr, savename=None, show=True, xlabel=r'$x$', ylabel=r'$y$', errorbar_kwargs={'fmt':'.', 'markersize':1, 'zorder':1, 'capsize':1, 'elinewidth':0.5, 'color':'k', 'alpha':1}):
    """
    Make the posterior predictive plot showing the 1, 2 and 3 sigma predictions
    of the function given the inferred parameters and plot the observed points on
    the same plot.
    
    Args:
        :reg (roxy.regressor.RoxyRegressor): The regressor object used for the inference
        :samples (dict): The MCMC samples, where the keys are the parameter names and values are ndarrays of the samples
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :savename (str, default=None): If not None, save the figure to the file given by this argument.
        :show (bool, default=True): If True, display the figure with plt.show()
        :xlabel (str, default='$x$'): The label to use for the x axis
        :ylabel (str, default='$x$'): The label to use for the y axis
        :errorbar_kwargs (dict): Dictionary of kwargs to pass to plt.errorbar
    
    Returns:
        :fig (matplotlib.figure.Figure): The figure containing the posterior predictive plot
    """

    names, all_samples = roxy.mcmc.samples_to_array(samples)
    pidx = reg.get_param_index(names, verbose=False)
    
    def f(x, theta):
        t = reg.param_default
        t = t.at[pidx].set(theta[:len(pidx)])
        return reg.value(x, t)

    print('\nMaking posterior predictive plot')
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, **errorbar_kwargs)
    
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    cbar = plot_contours(f, x, all_samples, ax)
    cbar = plt.colorbar(cbar,ticks=[0,1,2,3])
    cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    fig.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, transparent=True)
    if show:
        plt.show()

    return fig
