import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import xarray as xr
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm as normal_distribution
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
currentDateAndTime = datetime.now()
time_stamp = currentDateAndTime.strftime("%Y%m%d%H%M%S")

import asvgp.basis as basis
from asvgp.gpr import GPR_kron
import gpflow



seed = 1997
num_train = 2_000_000
num_test = 100_000

# Performance Metrics
def MSE(truth, prediction):
    se = (truth - prediction)**2
    mse = se.mean()
    return mse

def NLL(truth, pred_mean, pred_var):
    pred_std = np.sqrt(pred_var)
    nll = -normal_distribution.logpdf(truth, loc=pred_mean, scale=pred_std)
    return nll.mean()


# Get data
print('Preparing Data...')
print('')
data = xr.open_mfdataset('/home/jake/Data/NOC/eNATL60/1h/eNATL60-BLB002_sossheig_1h_y2009m08d01.nc')
data = data.where(data.nav_lon>-75, drop=True)
data = data.where(data.nav_lon<-30, drop=True)
data = data.where(data.nav_lat>20, drop=True)
data = data.where(data.nav_lat<50, drop=True)

# Preapare data
ssh_data = data.sossheig[0]
ssh = ssh_data.to_numpy().reshape(-1)
lat = ssh_data.nav_lat.to_numpy().reshape(-1)
lon = ssh_data.nav_lon.to_numpy().reshape(-1)
idx = (lon > -75) & (lon < -30) & (lat > 20) & (lat < 50) & (np.isnan(ssh) == False)
lon_clean = lon[idx]
lat_clean = lat[idx]
ssh_clean = ssh[idx]

# Make training and test data
seed = 1997
num_train = 2_000_000
num_test = 100_000

np.random.seed(seed)
idxs = np.arange(ssh_clean.reshape(-1).shape[0])
np.random.shuffle(idxs)
train_idx = idxs[:num_train]
test_idx = idxs[num_train:num_train+num_test]

lon_train = lon_clean[train_idx]
lat_train = lat_clean[train_idx]
ssh_train = ssh_clean[train_idx]
X_train = np.stack([lon_train, lat_train], axis=1)
y_train = ssh_train.reshape(-1,1)

lon_test = lon_clean[test_idx]
lat_test = lat_clean[test_idx]
ssh_test = ssh_clean[test_idx]
X_test = np.stack([lon_test, lat_test], axis=1)
y_test = ssh_test.reshape(-1,1)

# ASVGP
print('Training ASVGP...')
kernels = [gpflow.kernels.Matern32(), gpflow.kernels.Matern32()]
bases = [basis.B4Spline(-80, -25, 100), basis.B4Spline(15, 55, 100)]
t0 = time.time()
model_kron = GPR_kron((X_train, y_train), kernels, bases)
t_precomp = time.time() - t0
opt = gpflow.optimizers.Scipy()
t1 = time.time()
opt_logs = opt.minimize(model_kron.training_loss, variables=model_kron.trainable_variables)
t_opt = time.time() - t1
t_total = time.time() - t0
print("\n Time Precompute: {:.3e} \n Time Optimise: {:.3e} \n Time Total: {:.3e}".format(t_precomp, t_opt, t_total))

# Batch predict in 10_000 chunks
mean = np.zeros((num_test,1))
var = np.zeros((num_test,1))
for i in range(int(num_test / 10_000)):
    x_t = X_test[i*10_000:(i+1)*10_000,:]
    mean_, var_ = model_kron.predict_f(x_t)
    mean[i*10_000 : (i+1)*10_000] = mean_
    var[i*10_000 : (i+1)*10_000] = var_

# Report and save results
mse = MSE(y_test, mean)
nll = NLL(y_test, mean, var)
print("\n Mean squared error: {:.3e} \n Negative log-likelihood: {:.3e}".format(mse, nll))

metrics_index = pd.Index(['num_train', 'num_test', 'spline_order', 'time_precomp', 'time_opt', 'time_total', 'nlpd', 'mse', 'GP'])
experiment_index = pd.Index(['ASVGP'])
metrics = pd.DataFrame(dtype=float, index=experiment_index, columns=metrics_index)

metrics.loc['ASVGP', 'num_train'] = num_train
metrics.loc['ASVGP', 'num_test'] = num_test
metrics.loc['ASVGP', 'spline_order'] = model_kron.order
metrics.loc['ASVGP', 'time_precomp'] = t_precomp
metrics.loc['ASVGP', 'time_opt'] = t_opt
metrics.loc['ASVGP', 'time_total'] = t_total
metrics.loc['ASVGP', 'nll'] = nll
metrics.loc['ASVGP', 'mse'] = mse
metrics.loc['ASVGP', 'GP'] = model_kron

metrics.to_pickle('results/ASVGP_' + time_stamp + '.pkl')
print("\n", metrics)

# Figures
fig = plt.figure(figsize=(9,4))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='black', facecolor='white'))
interp = ax.contourf(lon_plot, lat_plot, mean_grid.reshape(k,k), 100, transform=ccrs.PlateCarree(), vmin=-1, vmax=1, cmap=cmocean.cm.balance)
interp.set_clim(-1,1)
plt.colorbar(interp, label='Sea Surface Height (m)', pad=0.02)
interp.set_clim(-1,1)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='white', alpha=0.5, linestyle='dotted')
gl.xlocator = mticker.FixedLocator([-70, -60, -50, -40, -30])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('eNATL60 - AS-VGP Predictive Mean')
plt.savefig('figures/ASVGP_mean_' + time_stamp + '.png', dpi=500, bbox_inches='tight')


