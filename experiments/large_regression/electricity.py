import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
import time
from asvgp.gpr import GPR_1d
import pandas as pd
from dateutil import parser
from sklearn.model_selection import train_test_split
from VFF.gpr import GPR_1d as VFF_gpr_1d
from VFF.vgp import VGP_1d as VFF_vgp_1d
import asvgp.basis as basis
from asvgp.gpr import GPR_1d
from sklearn.metrics import mean_squared_error
from tqdm.autonotebook import tqdm
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Inputs
m = 100
a = -1
b = m+1

def diff_time(date):
    return (parser.parse(date) - parser.parse(data['Date_time'][0])).total_seconds()

# Read input data
#data = pd.read_csv('/home/jake/Documents/ActuallySparseSVGP/data/household_power_consumption.txt', sep=';')
data = pd.read_pickle('/home/jake/Documents/ActuallySparseSVGP/data/household_power_consumption_dataframe.pkl')
data['Date_seq'] = (data['Date_seq'] - data['Date_seq'].min()) / (data['Date_seq'].max() - data['Date_seq'].min()) * m

# # Remove missing values
# data = (data[data['Global_active_power'] != '?'])
# data['Global_active_power'] = pd.to_numeric(data['Global_active_power'])

# # Convert time to continuous value
# data['Date_time'] = data['Date'] + str(' ') + data['Time']
# data['Date_seq'] = data['Date_time'].apply(diff_time)

# # Normalise inputs and outputs
# data['Global_active_power'] = (data['Global_active_power'] - data['Global_active_power'].mean()) / data['Global_active_power'].std()
# data['Date_seq'] = (data['Date_seq'] - data['Date_seq'].min()) / (data['Date_seq'].max() - data['Date_seq'].min()) * m

# data.to_pickle('/home/jake/Documents/ActuallySparseSVGP/data/household_power_consumption_dataframe.pkl')

def run_adam(model, train_dataset, minibatch_size, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf

svgp_nlpd = []
svgp_mse = []
svgp_opt_time = []
svgp_pred_time = []

vff_stoch_nlpd = []
vff_stoch_mse = []
vff_stoch_opt_time = []
vff_stoch_pred_time = []

band_nlpd = []
band_mse = []
band_opt_time = []
band_pred_time = []
band_total_time = []

print('Running experiments...')
print('')

for m in [1000]:

    print('Number of Inducing points = {}'.format(m))
    print('')

    for i in range(5):

        print('=> Iteration {}'.format(i))

        # Randomly select 90% for training and 10% for testing
        train, test = train_test_split(data, test_size=0.05, random_state=i)

        X = train['Date_seq'].to_numpy().reshape(-1,1)
        y = train['Global_active_power'].to_numpy().reshape(-1,1)

        X_test = test['Date_seq'].to_numpy().reshape(-1,1)
        y_test = test['Global_active_power'].to_numpy().reshape(-1,1)

        train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).repeat().shuffle(len(X))

        # SVGP
        t1a = time.time()
        kernel = gpflow.kernels.Matern32()
        likelihood = gpflow.likelihoods.Gaussian()
        Z = np.linspace(a, b, m).reshape(-1,1)
        model1 = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=len(X))
        opt = run_adam(model1, train_dataset, 100, 10_000)
        t1b = time.time()
        y_pred = np.zeros([X_test.shape[0], 1])
        for k in range(0, X_test.shape[0], 100_000):
            y_pred[k:k+100_000], _ = model1.predict_y(X_test[k:k+100_000])
        t1c = time.time()
        svgp_nlpd.append(-np.mean(model1.predict_log_density((X_test, y_test))))
        svgp_mse.append(mean_squared_error(y_test, y_pred))
        svgp_opt_time.append(t1b - t1a)
        svgp_pred_time.append(t1c - t1b)


        # Band - GPR
        t4a = time.time()
        kernel = gpflow.kernels.Matern52()
        splines = basis.B3Spline(a, b, m)
        model4 = GPR_1d((X,y), kernel, splines)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model4.training_loss, variables=model4.trainable_variables)
        t4b = time.time()
        y_pred, _ = model4.predict_f(X_test)
        t4c = time.time()
        band_nlpd.append(-np.mean(model4.predict_log_density((X_test, y_test)).numpy()))
        band_mse.append(mean_squared_error(y_test, y_pred))
        band_opt_time.append(t4b - t4a)
        band_pred_time.append(t4c - t4b)
        band_total_time.append(t4c - t4a)


        results = {
        'Model' : ['SVGP', 'Band'],
        'NLPD (mean)' : [np.mean(svgp_nlpd), np.mean(band_nlpd)],
        'NLPD (std)' : [np.std(svgp_nlpd), np.std(band_nlpd)],
        'MSE (mean)' : [np.mean(svgp_mse), np.mean(band_mse)],
        'MSE (std)' : [np.std(svgp_mse), np.std(band_mse)],
        'Time opt (mean)' : [np.mean(svgp_opt_time), np.mean(band_opt_time)],
        'Time pred (mean)' : [np.mean(svgp_pred_time), np.mean(band_pred_time)]
        }
        df = pd.DataFrame(results)
        print('')
        print(df)
        print('')

        

    results = {
        'Model' : ['SVGP', 'Band'],
        'NLPD (mean)' : [np.mean(svgp_nlpd), np.mean(band_nlpd)],
        'NLPD (std)' : [np.std(svgp_nlpd), np.std(band_nlpd)],
        'MSE (mean)' : [np.mean(svgp_mse), np.mean(band_mse)],
        'MSE (std)' : [np.std(svgp_mse), np.std(band_mse)],
        'Time opt (mean)' : [np.mean(svgp_opt_time), np.mean(band_opt_time)],
        'Time pred (mean)' : [np.mean(svgp_pred_time), np.mean(band_pred_time)]
    }

    df = pd.DataFrame(results)
    print('')
    print(df)
    print('')


