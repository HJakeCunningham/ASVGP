import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
import time
import asvgp.basis as basis
from asvgp.gpr import GPR_1d

# Load dataset
X = np.loadtxt('data/train_inputs').reshape(-1,1)
X_test = np.loadtxt('data/test_inputs').reshape(-1,1)
y = np.loadtxt('data/train_outputs').reshape(-1,1)
data = (X, y)

# Full GP
kernel = gpflow.kernels.Matern32()
gp = gpflow.models.GPR(data, kernel)
opt = gpflow.optimizers.Scipy()
opt.minimize(gp.training_loss, variables=gp.trainable_variables)
print('GP: ELBO = ',gp.maximum_log_likelihood_objective().numpy())

# ASVGP
a = -3.5                                    # Left edge of domain
b = 10.5                                    # Right edge of domain
m = 100                                     # Number of inducing variable
kernel = gpflow.kernels.Matern32()          # Matern kernel
splines = basis.B3Spline(a, b, m)           # B-Spline basis
asvgp = GPR_1d(data, kernel, splines)
opt = gpflow.optimizers.Scipy()
opt.minimize(asvgp.training_loss, variables=asvgp.trainable_variables)
print('ASVGP: ELBO = ',asvgp.maximum_log_likelihood_objective().numpy())

