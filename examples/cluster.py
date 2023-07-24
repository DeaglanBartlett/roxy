import numpy as np
import pandas as pd
from prettytable import PrettyTable
import jax.numpy as jnp
import matplotlib.pyplot as plt
from roxy.regressor import RoxyRegressor
import roxy.plotting
import roxy.mcmc
from getdist import plots, MCSamples

radius = 1.5

data = pd.read_csv('cluster_data.txt', delim_whitespace=True)
print(data.keys())

# Check the data is as expected
for s in ['sigma2001p5', 'sigma2001', 'Mdyn1p5', 'Mdyn1', 'MSZ']:
    m1 = (data[s] == '-')
    m2 = (data['err_' + s] == '-')
    assert m1.sum() == m2.sum(), "Missing entries do not match with corresponding errors"

# Replicate Table 1
ptab = PrettyTable()
ptab.field_names = ["Data set", "PSZ-North", "Others in PSZ2", "Beyond PSZ2", "1.5 x r200", "1 x r200"]
t0, t1, t2, t3, t4 = 0, 0, 0, 0, 0
for dset in ['LP15', 'ITP13', 'SDSS']:
    m0 = (data['Dataset'] == dset) & (data['PSZ_other'] == 0) & (data['Beyond_PSZ2'] == 0)
    m1 = (data['Dataset'] == dset) & (data['PSZ_other'] == 1) & (data['Beyond_PSZ2'] == 0)
    m2 = (data['Dataset'] == dset) & (data['PSZ_other'] == 0) & (data['Beyond_PSZ2'] == 1)
    m3 = (data['Dataset'] == dset)  & (data['Beyond_PSZ2'] == 0) & ((data['Flag'] == '2') | (data['Flag'] == '1'))
    m4 = (data['Dataset'] == dset) & (data['Flag'] == '1') & (data['Beyond_PSZ2'] == 0)
    print(m3.sum(), m4.sum())
    t0 += m0.sum()
    t1 += m1.sum()
    t2 += m2.sum()
    t3 += m3.sum()
    t4 += m4.sum()
    ptab.add_row([dset, m0.sum(), m1.sum(), m2.sum(), m3.sum(), m4.sum()])
ptab.add_row(['Total', t0, t1, t2, t3, t4])
print('\nCompare to Table 1 of arXiv:2111.13071')
print(ptab)

m = None
if radius == 1.5:
    m = (data['Flag'] == '2') | (data['Flag'] == '1')
    s = '1p5'
elif radius == 1:
    s = '1'
    m = (data['Flag'] == '1') & (data[f'Mdyn{s}'] != '-')
m &= (data['Beyond_PSZ2'] == 0)
print(m.sum())

data = data[m]

# Linear space
if radius == 1.5:
    s = '1p5'
elif radius == 1:
    s = '1'
conv_factor = 5/3  # = 10^{15} / (6 x 10^{14}) = 10 / 6 = 5 / 3
xobs = data[f'Mdyn{s}'].to_numpy(float) * conv_factor
yobs = data['MSZ'].to_numpy(float) * conv_factor
xerr = data[f'err_Mdyn{s}'].to_numpy(float) * conv_factor
yerr = data['err_MSZ'].to_numpy(float) * conv_factor

# Log space
xerr = jnp.array(xerr / xobs)
yerr = jnp.array(yerr / yobs)
xobs = jnp.array(jnp.log(xobs))
yobs = jnp.array(jnp.log(yobs))

print('Average x error:', np.mean(xerr), np.median(xerr))
print('Average y error:', np.mean(yerr), np.median(yerr))

# Define function
def my_fun(x, theta):
    return theta[0] * x + theta[1]
param_names = ['alpha', 'c']
theta0 = [1.0, 0.5]  # defaults
param_prior = {'alpha':[-3, 3], 'c':[-5, 5], 'sig':[0, 3]}

reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

# MCMC params
nwarm = 700
nsamp = 5000

all_method = ['mnr', 'unif', 'prof']
all_method_label = ['MNR', 'Uniform', 'Profile']

ranges = param_prior
ranges['w_gauss'] = [0, None]
all_samps = []
all_gradient = []
all_intercept = []

for method, method_label in zip(all_method, all_method_label):
    print('\nMETHOD:', method)
    samps = reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method)
    
    if method == 'mnr':
        mnr_samps = samps.copy()
    
    alpha = samps['alpha']
    print('alpha: %.3f +/- %.3f %.3f'%(np.median(alpha), np.percentile(alpha, 84) - np.median(alpha), np.median(alpha) - np.percentile(alpha, 16)))
    mB = np.exp(samps['c'])
    print('1 - B: %.3f +/- %.3f %.3f'%(np.median(mB), np.percentile(mB, 84) - np.median(mB), np.median(mB) - np.percentile(mB, 16)))
    all_gradient.append(np.median(alpha))
    all_intercept.append(np.median(samps['c']))
    
    names, samps = roxy.mcmc.samples_to_array(samps)
    
    labs = list(names)
    for p, l in zip(['mu_gauss', 'w_gauss', 'sig'], [r'\mu', r'w', r'\sigma_{\rm int}']):
        if (p in names):
            i = np.squeeze(np.where(names==p))
            labs[i] = l
    i = np.squeeze(np.where(names=='alpha'))
    labs[i] = r'$\alpha$'
                
    all_samps.append(MCSamples(
            samples=samps,
            names=names,
            labels=labs,
            ranges=ranges,
            label=method_label
        ))
    all_samps[-1].addDerived(np.exp(all_samps[-1]['c']), name='mB', label='1-B')

g = plots.get_subplot_plotter(width_inch=5)
g.triangle_plot(all_samps, ['alpha', 'mB', 'sig', 'mu_gauss', 'w_gauss'], filled=True)
plt.gcf().align_labels()
plt.savefig('cluster_corner.pdf')
plt.clf()
plt.close(plt.gcf())

errorbar_kwargs={'fmt':'.', 'markersize':1, 'zorder':-1, 'capsize':1, 'elinewidth':0.2, 'color':'k', 'alpha':1}
fig = roxy.plotting.posterior_predictive_plot(reg, mnr_samps, xobs, yobs, xerr, yerr, savename=None, show=False, xlabel=r'$\log \left( \frac{M_{500}^{\rm dyn}}{6 \times 10^{14} {\rm \, M_{\odot}}} \right)$', ylabel=r'$\log \left( \frac{M_{500}^{\rm SZ}}{6 \times 10^{14} {\rm \, M_{\odot}}} \right)$', errorbar_kwargs=errorbar_kwargs)
ax = fig.gca()
x = np.array(ax.get_xlim())
for gradient, intercept, method_label in zip(all_gradient, all_intercept, all_method_label):
    ax.plot(x, gradient * x + intercept, label=method_label)
ax.legend()
ax.set_xlim(x)
plt.savefig('cluster_predictive.pdf')
plt.clf()
plt.close(fig)
