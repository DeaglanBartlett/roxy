import corner
import numpy as np
import matplotlib.pyplot as plt
import roxy.mcmc
from getdist import plots, MCSamples
import arviz as az

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
    
# Plot data and the function
