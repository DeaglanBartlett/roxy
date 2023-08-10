import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import warnings
from roxy.regressor import RoxyRegressor
import roxy.plotting
import roxy.mcmc
import scipy.stats
import os

plt.rc('text', usetex=False)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(4)

which_run = 'new'
repeat_fit = False
nwarm, nsamp = 700, 10000
nrepeat = 150
max_ngauss = 4

# Parameters which vary
# old: [Atrue, sig_true, Npoints, xerr_mean, exp_scale]
# new: [sig_true, Npoints, Atrue, xerr_mean, exp_scale]
if which_run == 'old':
    all_param = [
        [15.0, 0.0, 10, 20.0, 15.0],  # Figure 5
        [-15.0, 10.0, 894, 20.0, 15.0],
        [15.0, 0.0, 4000, 15.0, 15.0],
        [15.0, 0.0, 4000, 20.0, 15.0],
        [-15.0, 5.0, 4000, 20.0, 15.0],
        [15.0, 0.0, 10, 20.0, 15.0],
        [15.0, 10.0, 4000, 5.0, 15.0],
        [2, 1000, 5, 2.3, 8],
        [2, 1000, 5, 20, 8],
    ]
    fig5_idx = 0
elif which_run == 'new':
    all_param = [
        [5.0, 4000, -30.0, 20.0, 15.0],
        [15.0, 4000, 30.0, 20.0, 15.0],
        [15.0, 4000, 30.0, 20.0, 15.0],
        [0.0, 4000, -15.0, 20.0, 15.0],
        [10.0, 10, -30.0, 20.0, 15.0], # Figure 5
        [10.0, 45, 0.0, 20.0, 15.0],
    ]
    fig5_idx = 4

# Fixed parameters
Btrue = 1.
yerr_mean = 2
yerr_std = 0.2

# Divide repeats among ranks
rank_nrepeat = nrepeat // size
remainder = nrepeat - size * rank_nrepeat
if rank < remainder:
    rank_nrepeat += 1

def my_fun(x, theta):
    return theta[0] * x + theta[1]
param_names = ['A', 'B']
param_prior = {'A':[None, None], 'B':[None, None], 'sig':[None, None]}

for ipar, par in enumerate(all_param):

    if rank == 0:
        print(f'Parameter set {ipar+1} of {len(all_param)}', flush=True)

    if repeat_fit:
        if which_run == 'old':
            Atrue, sig_true, Npoints, xerr_mean, exp_scale = par
        elif which_run == 'new':
            sig_true, Npoints, Atrue, xerr_mean, exp_scale = par
        theta0 = [Atrue, Btrue]
        xerr_std = xerr_mean / 5

        reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)
        
        all_bias = np.empty((max_ngauss, 3, rank_nrepeat))
        all_ic = np.empty((max_ngauss, 2, rank_nrepeat))  # (nll, BIC)
        
        np.random.seed(rank)
        
        for i in range(rank_nrepeat):
        
            if rank == 0:
                print(f'\t{i+1} of {rank_nrepeat}', flush=True)

            # Make data
            xtrue = np.random.exponential(exp_scale, Npoints)
            ytrue = reg.value(xtrue, theta0)
            xerr = np.random.normal(xerr_mean, xerr_std, Npoints)
            xerr[xerr<0.]=0.
            yerr = np.random.normal(yerr_mean, yerr_std, Npoints)
            yerr[yerr<0.]=0.
            xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
            yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig_true ** 2)
            
            for ngauss in range(1, max_ngauss+1):
            
                if rank == 0:
                    print(f'\t\tngauss = {ngauss}', flush=True)
            
                if ngauss == 1:
                    kwargs = {'method':'mnr'}
                else:
                    kwargs = {'method':'gmm', 'gmm_prior':'hierarchical', 'ngauss':ngauss}
                kwargs['seed'] = 1234
                kwargs['progress_bar'] = False
                kwargs['verbose'] = False

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    samples = reg.mcmc(
                                param_names,
                                xobs,
                                yobs,
                                [xerr, yerr],
                                nwarm,
                                nsamp,
                                **kwargs
                            )

                truths = {'A':Atrue, 'B':Btrue, 'sig':sig_true}

                try:
                    biases = roxy.mcmc.compute_bias(samples, truths, verbose=False)
                    all_bias[ngauss-1,0,i] = biases['A']
                    all_bias[ngauss-1,1,i] = biases['B']
                    all_bias[ngauss-1,2,i] = biases['sig']
                except:
                    all_bias[ngauss-1,:,i]
                
                all_ic[ngauss-1,:,i] = reg.compute_information_criterion(
                        'BIC',
                        param_names,
                        xobs,
                        yobs,
                        [xerr,yerr],
                        nwarm=100,
                        nsamp=100,
                        **kwargs
                        )
        
        all_bias = comm.gather(all_bias, root=0)
        all_ic = comm.gather(all_ic, root=0)
        if rank == 0:
            all_bias = np.concatenate(all_bias, axis=2)
            all_ic = np.concatenate(all_ic, axis=2)
            for ngauss in range(1, max_ngauss+1):
                np.savez(f'bias_res_{ipar}_{ngauss}.npz',
                    bias=all_bias[ngauss-1,...],
                    ic=all_ic[ngauss-1,...]
                )
        
    comm.Barrier()

    if rank == 0:

        if not os.path.isdir('figs'):
            os.mkdir('figs')
        
        cm = plt.get_cmap('Set1')
        fig, axs = plt.subplots(1, max_ngauss, figsize=(15,4), sharex=True)
        for ngauss in range(1, max_ngauss+1):
            bias = np.load(f'bias_res_{ipar}_{ngauss}.npz')['bias']
            m = np.isfinite(bias)
            m = np.prod(m, axis=0).astype(bool)
            bias = bias[:,m]
            bic = np.load(f'bias_res_{ipar}_{ngauss}.npz')['ic'][1,m]
            sigmax = 10
            xx = np.linspace(-sigmax, sigmax, 200)
            for i, label in enumerate([r'$A$', r'$B$', r'$\sigma_{\rm int}$']):
                m = (bias[i,:] >= -sigmax) & (bias[i,:] <= sigmax)
                kde = scipy.stats.gaussian_kde(bias[i,m])
                kde = kde(xx)
                if i >= 2:
                    c = cm(i)
                else:
                    c = cm(1-i)
                axs[ngauss-1].plot(xx, kde, color=c, label=label)
            
            axs[ngauss-1].axvline(x=0, color='k')
            axs[ngauss-1].legend()
            axs[ngauss-1].set_xlabel('Bias')
            axs[ngauss-1].set_ylim(0, None)
            axs[ngauss-1].get_yaxis().set_ticklabels([])
            title = f'{ngauss} Gaussian'
            if ngauss > 1:
                title += 's'
            axs[ngauss-1].set_title(title)
            
        fig.tight_layout()
        fig.savefig(f'figs/bias_res_{ipar}.png')
        fig.clf()
        plt.close(fig)



if rank == 0:

    nwarm, nsamp = 5000, 10000

    np.random.seed(rank)

    par = all_param[fig5_idx]

    if which_run == 'old':
        Atrue, sig_true, Npoints, xerr_mean, exp_scale = par
    elif which_run == 'new':
        sig_true, Npoints, Atrue, xerr_mean, exp_scale = par
    theta0 = [Atrue, Btrue]
    xerr_std = xerr_mean / 5

    reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

    # Make data
    xtrue = np.random.exponential(exp_scale, Npoints)
    ytrue = reg.value(xtrue, theta0)
    xerr = np.random.normal(xerr_mean, xerr_std, Npoints)
    xerr[xerr<0.]=0.
    yerr = np.random.normal(yerr_mean, yerr_std, Npoints)
    yerr[yerr<0.]=0.
    xobs = xtrue + np.random.normal(size=len(xtrue)) * xerr
    yobs = ytrue + np.random.normal(size=len(xtrue)) * np.sqrt(yerr ** 2 + sig_true ** 2)

    samples = reg.mcmc(
                    param_names,
                    xobs,
                    yobs,
                    [xerr, yerr],
                    nwarm,
                    nsamp,
                    method='mnr',
                    seed=1234,
                    progress_bar=True,
                    verbose=True
                )
    roxy.plotting.triangle_plot(samples, 
            to_plot='all', 
            module='getdist', 
            param_prior=param_prior, 
            show=False,
            savename='maxbias_corner.png',
    )
    roxy.plotting.trace_plot(
            samples, 
            to_plot='all',
            show=False,
            savename='maxbias_trace.png',
    )

