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
    else:
        labs = [labels[n] for n in names]
        
    if module == 'corner':
        corner.corner(all_samples, labels=labs)
    elif module == 'getdist':
    
        if param_prior is None:
            ranges = {}
        else:
            ranges = param_prior
        ranges['w_gauss'] = [0, None]
    
        samps = MCSamples(
            samples=all_samples,
            names=names,
            labels=labs,
            ranges=ranges
        )
            
        g = plots.get_subplot_plotter()
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

    res = az.from_dict(samples)
    if to_plot == 'all':
        az.plot_trace(res, compact=True)
    else:
        az.plot_trace(res, compact=True, var_names=to_plot)
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
