import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from roxy.regressor import RoxyRegressor
import roxy.plotting
import roxy.mcmc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(4)

# Parameter which vary
# [Atrue, sig_true, Npoints, xerr_mean, exp_scale]
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

# Fixed parameters
Btrue = 1.
yerr_mean = 2
yerr_std = 0.2

# MCMC params
nwarm, nsamp = 700, 500
#nwarm, nsamp = 700, 10000
nrepeat = 10
max_ngauss = 2
repeat_fit = False

# Divide repeats among ranks
rank_nrepeat = nrepeat // size
remainder = nrepeat - size * rank_nrepeat
if rank < remainder:
    rank_nrepeat += 1

def my_fun(x, theta):
    return theta[0] * x + theta[1]
param_names = ['A', 'B']

#param_prior = {'A':[0, 50], 'B':[-50, 50], 'sig':[0, 100]}
#param_prior = {'A':[-200, 200], 'B':[-1000, 1000], 'sig':[0, 600]}
param_prior = {'A':[None, None], 'B':[None, None], 'sig':[None, None]}

for ipar, par in enumerate(all_param[:1]):

    if repeat_fit:
        Atrue, sig_true, Npoints, xerr_mean, exp_scale = par
        theta0 = [Atrue, Btrue]
        xerr_std = xerr_mean / 5

        reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)
        
        all_bias = np.empty((max_ngauss, 3, rank_nrepeat))
        
        np.random.seed(rank)
        
        for i in range(rank_nrepeat):

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
            
                if ngauss == 1:
                    kwargs = {'method':'mnr'}
                else:
                    kwargs = {'method':'gmm', 'gmm_prior':'hierarchical', 'ngauss':ngauss}
                kwargs['seed'] = 1234
                kwargs['progress_bar'] = False

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
                biases = roxy.mcmc.compute_bias(samples, truths)

                all_bias[ngauss-1,0,i] = biases['A']
                all_bias[ngauss-1,1,i] = biases['B']
                all_bias[ngauss-1,2,i] = biases['sig']
        
        all_bias = comm.gather(all_bias, root=0)
        if rank == 0:
            all_bias = np.concatenate(all_bias, axis=2)
            np.save(f'bias_res_{ipar}.npy', all_bias)
        
    comm.Barrier()

    if rank == 0:

        all_bias = np.load(f'bias_res_{ipar}.npy')
        
        cm = plt.get_cmap('Set1')
        fig, axs = plt.subplots(1, max_ngauss, figsize=(10,4), sharex=True)
        for ngauss in range(1, max_ngauss+1):
            for i, label in enumerate([r'$A$', r'$B$', r'$\sigma_{\rm int}$']):
                axs[ngauss-1].hist(
                        all_bias[ngauss-1,i,:],
                        color=cm(i),
                        bins=10,
                        histtype='step',
                        density=True,
                        label=label,
                )
            axs[ngauss-1].legend()
            axs[ngauss-1].set_xlabel('Bias')
            title = f'{ngauss} Gaussian'
            if ngauss > 1:
                title += 's'
            axs[ngauss-1].set_title(title)
            
        fig.tight_layout()
        plt.show()
