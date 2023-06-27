import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import roxy.mcmc
from getdist import plots, MCSamples
import arviz as az
from fgivenx import plot_contours

rcParams['text.usetex'] = True

def triangle_plot(samples, labels=None, to_plot='all', module='corner', param_prior=None, savename=None, show=True):

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
        
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())
    
    return
    
    
def trace_plot(samples, labels=None, to_plot='all', savename=None, show=True):

    res = az.from_dict(samples)
    if to_plot == 'all':
        az.plot_trace(res, compact=True)
    else:
        az.plot_trace(res, compact=True, var_names=to_plot)
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())

    return
    
def posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr, savename=None, show=True):

    names, all_samples = roxy.mcmc.samples_to_array(samples)
    pidx = reg.get_param_index(names, verbose=False)
    
    def f(x, theta):
        # Parameters of function
        t = reg.param_default
        t = t.at[pidx].set(theta[:len(pidx)])
        return reg.value(x, t)
        
    x = np.linspace(xobs.min(), xobs.max(), 200)
    plot_kwargs = {'fmt':'.', 'markersize':1, 'zorder':1,
                 'capsize':1, 'elinewidth':0.5, 'color':'k', 'alpha':1}
        
    print('\nMaking posterior predictive plot')
    fig, ax = plt.subplots(1, 1)
    cbar = plot_contours(f, x, all_samples, ax)
    cbar = plt.colorbar(cbar,ticks=[0,1,2,3])
    cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])
    ax.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, **plot_kwargs)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    fig.tight_layout()
    
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())

    return
