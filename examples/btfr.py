import numpy as np
import roxy.plotting
import roxy.mcmc
from roxy.regressor import RoxyRegressor
from getdist import plots, MCSamples
import matplotlib.pyplot as plt

# Get data
arr = np.genfromtxt("BTFR.dat", dtype=None, encoding=None)
mask = arr['f5']!=0
print("BTFR:", len(arr), len(arr[mask]))
arr = arr[mask]
yobs, yerr = arr['f1'], arr['f2']                                              # log(Mbar)
xobs, xerr = np.log10(arr['f5']), arr['f6']/(arr['f5']*np.log(10.))            # log(Vflat)

print('Average x error:', np.mean(xerr), np.median(xerr))
print('Average y error:', np.mean(yerr), np.median(yerr))

# Define function
def my_fun(x, theta):
    return theta[0] * x + theta[1]
param_names = ['A', 'B']
theta0 = [2, 0.5]  # defaults
param_prior = {'A':[-50, 50], 'B':[-50, 50], 'sig':[0, 30]}

reg = RoxyRegressor(my_fun, param_names, theta0, param_prior)

# MCMC params
nwarm = 700
nsamp = 5000
method = 'mnr'

ranges = param_prior
ranges['w_gauss'] = [0, None]
all_samps = []
all_gradient = []
all_intercept = []

all_method = ['mnr', 'uniform', 'profile']
all_method_label = ['MNR', 'Uniform', 'Profile']

for method, method_label in zip(all_method, all_method_label):
    samps = reg.mcmc(param_names, xobs, yobs, [xerr, yerr], nwarm, nsamp, method=method)
    
    if method == 'mnr':
        mnr_samps = samps.copy()
        
    all_gradient.append(np.median(samps['A']))
    all_intercept.append(np.median(samps['B']))
        
    names, samps = roxy.mcmc.samples_to_array(samps)
    
    labs = list(names)
    for p, l in zip(['mu_gauss', 'w_gauss', 'sig'], [r'\mu', r'w', r'\sigma_{\rm int}']):
        if (p in names):
            i = np.squeeze(np.where(names==p))
            labs[i] = l
                
    all_samps.append(MCSamples(
            samples=samps,
            names=names,
            labels=labs,
            ranges=ranges,
            label=method_label
        ))

g = plots.get_subplot_plotter(width_inch=5)
g.triangle_plot(all_samps, ['A', 'B', 'sig', 'mu_gauss', 'w_gauss'], filled=True)
plt.savefig('btfr_corner.pdf')
plt.clf()
plt.close(plt.gcf())

xlabel = r'$\log_{10} \left( \frac{V_{\rm flat}}{{\rm km \, s^{-1}}} \right)$'
ylabel = r'$\log_{10} \left( \frac{M_{\rm bar}}{{\rm \, M_{\odot}}} \right)$'

errorbar_kwargs={'fmt':'.', 'markersize':1, 'zorder':-1, 'capsize':1, 'elinewidth':0.2, 'color':'k', 'alpha':1}
fig = roxy.plotting.posterior_predictive_plot(reg, mnr_samps, xobs, yobs, xerr, yerr, savename=None, show=False, xlabel=xlabel, ylabel=ylabel, errorbar_kwargs=errorbar_kwargs)
ax = fig.gca()
x = np.array(ax.get_xlim())
for gradient, intercept, method_label in zip(all_gradient, all_intercept, all_method_label):
    ax.plot(x, gradient * x + intercept, label=method_label)
ax.legend()
ax.set_xlim(x)
plt.savefig('btfr_predictive.pdf')
plt.clf()
plt.close(fig)
    
