import warnings
import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
from fgivenx import plot_contours
from getdist import MCSamples, plots
from matplotlib import rcParams, MatplotlibDeprecationWarning

import roxy.mcmc

rcParams['text.usetex'] = False
rcParams.update({'font.size': 14})


def triangle_plot(samples, labels=None, to_plot='all', module='corner',
                  truths=None, param_prior=None, savename=None, show=True):
    """
    Plot the 1D and 2D posterior distributions of the parameters in a triangle plot.

    Args:
        :samples (dict): The MCMC samples, where the keys are the parameter names and
            values are ndarrays of the samples
        :labels (dict, default=None): Dictionary of parameter labels ot use in the plot.
            If None, then use the names given as keys in samples.
        :to_plot (list, default='all'): If 'all', then use all parameters. If a list,
            then only use the parameters given in that list
        :module (str, default='corner'): Which module to use to make the triangle plot
            ('corner' or 'getdist' currently available)
        :truths (dict, default=None): If not None, use this to specify the true
            values of the parameters to plot.
        :param_prior (dict, default=None): If not None and using 'getdist', use this to
            specify the range of the varibales to prevent undesirable smoothing effects.
        :savename (str, default=None): If not None, save the figure to the file given
            by this argument.
        :show (bool, default=True): If True, display the figure with plt.show()
    """

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            category=MatplotlibDeprecationWarning,
            module=r"getdist\.matplotlib_ext",
        )

        names, all_samples = roxy.mcmc.samples_to_array(samples)

        if to_plot != 'all':
            idx = [np.squeeze(np.where(names == p)) for p in to_plot]
            names = names[idx]
            all_samples = all_samples[:, idx]

        if labels is None:
            labs = list(names)
            for p, label in zip(['mu_gauss', 'w_gauss', 'sig'],
                                [r'\mu_{\rm gauss}', r'w_{\rm gauss}', r'\sigma_{\rm int}']):
                if (p in names) and ((p in to_plot) or (to_plot == 'all')):
                    i = np.squeeze(np.where(names == p))
                    labs[i] = label

            #  GMM parameters
            if 'weights_0' in names:
                #  Extract number of Gaussians
                ngauss = len([n for n in names if n.startswith('weights')])
                for i in range(ngauss):
                    for p, label in zip([f'mu_gauss_{i}', f'w_gauss_{i}', f'weights_{i}'],
                                        [f'\\mu_{i}', f'w_{i}', f'\\nu_{i}']):
                        j = np.squeeze(np.where(names == p))
                        labs[j] = label

            # Kelly prior parameters
            if 'hyper_mu' in names:
                for p, label in zip(['hyper_mu', 'hyper_w2', 'hyper_u2'],
                                    [r'\mu_\star', r'w_\star^2', r'u_\star^2']):
                    j = np.squeeze(np.where(names == p))
                    labs[j] = label

        else:
            labs = [labels[n] for n in names]

        if module == 'corner':
            labs = ['$' + label + '$' for label in labs]
            fig, _ = plt.subplots(len(labs), len(labs), figsize=(8, 8))
            markers = None
            if truths is not None:
                markers = [truths.get(n, None) for n in names]
            corner.corner(all_samples, labels=labs, fig=fig, truths=markers)
        elif module == 'getdist':

            if param_prior is None:
                ranges = {}
            else:
                ranges = param_prior
            ranges['w_gauss'] = [0, None]
            if ('sig' in ranges and
                    ((param_prior['sig'][0] is None) or (param_prior['sig'][1] is None))):
                ranges['sig'] = [0, param_prior['sig'][1]]

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
            g.triangle_plot(samps, filled=True, markers=truths)

        else:
            raise NotImplementedError
        plt.gcf().align_labels()

        if savename is not None:
            plt.savefig(savename, transparent=False)
        if show:
            plt.show()
        plt.clf()
        plt.close(plt.gcf())


def trace_plot(samples, to_plot='all', truths=None, savename=None, show=True):
    """
    Plot the trace of the parameter values as a function of MCMC step

    Args:
        :samples (dict): The MCMC samples, where the keys are the parameter names and
            values are ndarrays of the samples
        :to_plot (list, default='all'): If 'all', then use all parameters. If a list,
            then only use the parameters given in that list
        :truths (dict, default=None): If not None, use this to specify the true
            values of the parameters to plot.
        :savename (str, default=None): If not None, save the figure to the file given by
            this argument.
        :show (bool, default=True): If True, display the figure with plt.show()
    """

    samples = {key: np.asarray(value)[None, ...]
               if np.asarray(value).ndim == 1 else np.asarray(value)
               for key, value in samples.items()}

    # Check for GMM
    if 'weights' in samples:
        new_samples = samples.copy()
        for k in ['mu_gauss', 'w_gauss', 'weights']:
            new_samples.pop(k)
            v = samples[k]
            for i in range(v.shape[1]):
                component = v[:, i]
                if component.ndim == 1:
                    component = component[None, ...]
                new_samples[f'{k}_{i}'] = component
        npar = len(new_samples.keys())
        res = az.from_dict({'posterior': new_samples})
    else:
        res = az.from_dict({'posterior': samples})
        npar = len(samples.keys())

    if to_plot != 'all':
        npar = len(to_plot)
    figsize = (12, min(2 * npar, 10))

    plot_kwargs = {'figure_kwargs': {'figsize': figsize}}
    if to_plot != 'all':
        plot_kwargs['var_names'] = to_plot
    az.plot_trace(res, **plot_kwargs)

    if truths is not None:
        plotted_names = list(res['posterior'].data_vars)
        if to_plot != 'all':
            plotted_names = [name for name in plotted_names if name in to_plot]
        for axis, name in zip(plt.gcf().axes[::2], plotted_names):
            if name in truths:
                axis.axhline(truths[name], color='C1', linestyle='--')
    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename, transparent=False)
    if show:
        plt.show()
    plt.clf()
    plt.close(plt.gcf())


def posterior_predictive_plot(reg, samples, xobs, yobs, xerr, yerr, y_is_detected=None,
                              savename=None, show=True, xlabel=r'$x$', ylabel=r'$y$',
                              errorbar_kwargs=None,
                              fgivenx_kwargs=None, xscale='linear',
                              yscale='linear', xlim=None, ylim=None):
    """
    Make the posterior predictive plot showing the 1, 2 and 3 sigma predictions
    of the function given the inferred parameters and plot the observed points on
    the same plot.

    Args:
        :reg (roxy.regressor.RoxyRegressor): The regressor object used for the inference
        :samples (dict): The MCMC samples, where the keys are the parameter names and
            values are ndarrays of the samples
        :xobs (jnp.ndarray): The observed x values
        :yobs (jnp.ndarray): The observed y values
        :xerr (jnp.ndarray): The error on the observed x values
        :yerr (jnp.ndarray): The error on the observed y values
        :y_is_detected (array-like, default=[]): Boolean array of the same length as
            yobs, where True indicates a detected point and False indicates an upper limit.
        :savename (str, default=None): If not None, save the figure to the file given
            by this argument.
        :show (bool, default=True): If True, display the figure with plt.show()
        :xlabel (str, default='$x$'): The label to use for the x axis
        :ylabel (str, default='$x$'): The label to use for the y axis
        :errorbar_kwargs (dict): Dictionary of kwargs to pass to plt.errorbar
        :fgivenx_kwargs (dict): Dictionary of kwargs to pass to fgivenx.plot_contours
        :xscale (str, default='linear'): Scale to use for x axis ('linear' or 'log')
        :yscale (str, default='linear'): Scale to use for y axis ('linear' or 'log')
        :xlim (tuple, default=None): If not None, set the x limits to this value
        :ylim (tuple, default=None): If not None, set the y limits to this value

    Returns:
        :fig (matplotlib.figure.Figure): The figure containing the posterior predictive
            plot
    """

    if y_is_detected is None:
        y_is_detected = []
    if errorbar_kwargs is None:
        errorbar_kwargs = {'fmt': '.', 'markersize': 1,
                            'zorder': 10, 'capsize': 1,
                            'elinewidth': 0.5, 'color': 'k', 
                            'alpha': 1}

    names, all_samples = roxy.mcmc.samples_to_array(samples)
    pidx = reg.get_param_index(names, verbose=False)

    def f(x, theta):
        t = reg.param_default
        t = t.at[pidx].set(theta[:len(pidx)])
        return reg.value(x, t)

    print('\nMaking posterior predictive plot')
    fig, ax = plt.subplots(1, 1)
    if len(y_is_detected) > 0:
        ax.errorbar(xobs[y_is_detected], yobs[y_is_detected],
                    xerr=xerr, yerr=yerr, **errorbar_kwargs)
        ax.errorbar(xobs[~y_is_detected], yobs[~y_is_detected],
                    xerr=xerr, yerr=yerr, uplims=True, **errorbar_kwargs)
    else:
        ax.errorbar(xobs, yobs, xerr=xerr, yerr=yerr, **errorbar_kwargs)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    xmin, xmax = ax.get_xlim()
    if xscale == 'log':
        x = np.logspace(np.log10(xmin), np.log10(xmax), 200)
    else:
        x = np.linspace(xmin, xmax, 200)
    if fgivenx_kwargs is None:
        cbar = plot_contours(f, x, all_samples, ax)
    else:
        cbar = plot_contours(f, x, all_samples, ax, **fgivenx_kwargs)
    cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    fig.tight_layout()

    if savename is not None:
        plt.savefig(savename, transparent=False)
    if show:
        plt.show()

    return fig
