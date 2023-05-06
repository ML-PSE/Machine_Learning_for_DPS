##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                      FFNN-NARX-based Modeling of Heat Exchangers 
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import matplotlib.pyplot as plt, numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import ccf
from matplotlib.ticker import MaxNLocator

# package settings
plt.rcParams.update({'font.size': 14})

#%% read data and plot
data = np.loadtxt('exchanger.dat')
u = data[:,1, None]; y = data[:,2, None]

# plots
plt.figure(figsize=(16,4.5))
plt.plot(u, 'steelblue', linewidth=0.8, drawstyle='steps')
plt.ylabel('u(k)'), plt.xlabel('k'), plt.xlim(0)

plt.figure(figsize=(16,4.5))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y(k)'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

#%% split dataset into fitting and test dataset
u_fit = u[:3000,0:1]; u_test = u[3000:,0:1] 
y_fit = y[:3000,0:1]; y_test = y[3000:,0:1] 

#%% scale data before model fitting
u_scaler = StandardScaler(); u_fit_scaled = u_scaler.fit_transform(u_fit); u_test_scaled = u_scaler.transform(u_test) 
y_scaler = StandardScaler(); y_fit_scaled = y_scaler.fit_transform(y_fit); y_test_scaled = y_scaler.transform(y_test) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                        Add lagged variables as regressors
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rearrange u and y data into (# sequence samples, # lagged regressors) form
u_lags = 9; y_lags = 3
u_augument_fit_scaled = []
y_augment_fit_scaled = []

for sample in range(max(u_lags, y_lags), u_fit.shape[0]):
    row = np.hstack((u_fit_scaled[sample-u_lags:sample,0], y_fit_scaled[sample-y_lags:sample,0]))
    u_augument_fit_scaled.append(row)
    y_augment_fit_scaled.append(y_fit_scaled[sample])

# conversion: convert list into array 
u_augument_fit_scaled = np.array(u_augument_fit_scaled)
y_augment_fit_scaled = np.array(y_augment_fit_scaled)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          fit FFNN-NARX model
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import Keras libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#%% define model
model = Sequential()
model.add(Dense(14, activation='relu', kernel_initializer='he_normal', input_shape=(12,))) # 14 neurons in 1st hidden layer; this hidden layer accepts data from a 12 dimensional input
model.add(Dense(7, activation='relu', kernel_initializer='he_normal')) # 7 neurons in 2nd layer
model.add(Dense(1)) # output layer

#%% model summary
model.summary()

#%% compile and fit model
model.compile(loss='mse', optimizer='Adam') # mean-squared error is to be minimized
model.fit(u_augument_fit_scaled, y_augment_fit_scaled, epochs=250, batch_size=125)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         Residual analysis on training dataset
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% get model (1-step ahead)  predictions and residuals on training dataset
y_augment_fit_scaled_pred = model.predict(u_augument_fit_scaled) 
y_augment_fit_pred =  y_scaler.inverse_transform(y_augment_fit_scaled_pred)
y_augment_fit = y_scaler.inverse_transform(y_augment_fit_scaled)
residuals_fit = y_augment_fit - y_augment_fit_pred

plt.figure(figsize=(10,3.5)), plt.plot(y_augment_fit, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_augment_fit_pred, 'r', linewidth=0.8, label='Predictions')
plt.title('Training data'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(6,2.5)), plt.plot(residuals_fit, 'black', linewidth=0.8)
plt.title('Training data'), plt.ylabel('residuals'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

#%% ACF of residuals
conf_int = 1.96/np.sqrt(len(residuals_fit))

plot_acf(residuals_fit, lags= 40, alpha=None, title='')
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag'), plt.title('Autocorrelations'), plt.ylim(-0.2, 0.5) 
plt.show()

#%% CCF b/w residuals and input sequence
ccf_vals = ccf(residuals_fit, u_fit[u_lags:], adjusted=False) # ccf for lag > 0
ccf_vals = ccf_vals[1:41] # ccf for lag 1 to 40

# generate CCF plot
lags = np.arange(1,41)

plt.figure(figsize=(6,4))
plt.vlines(lags, [0], ccf_vals)
plt.axhline(0, 0, lags[-1])
plt.plot(lags, ccf_vals, marker='o', markersize=5, linestyle='None')
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.xlabel('lag'), plt.title('Cross-correlations')
plt.ylim(-0.5, 0.5)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##               m-step ahead predictions on test dataset
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% rearrange u and y data into (# sequence samples, # lagged regressors) form
u_augument_test_scaled = []
y_augment_test_scaled = []

for sample in range(max(u_lags, y_lags), u_test.shape[0]):
    row = np.hstack((u_test_scaled[sample-u_lags:sample,0], y_test_scaled[sample-y_lags:sample,0]))
    u_augument_test_scaled.append(row)
    y_augment_test_scaled.append(y_test_scaled[sample])

# conversion: convert list into array 
u_augument_test_scaled = np.array(u_augument_test_scaled)
y_augment_test_scaled = np.array(y_augment_test_scaled)

#%% 1-step ahead predictions
y_augment_test_scaled_pred = model.predict(u_augument_test_scaled) 
y_augment_test_pred =  y_scaler.inverse_transform(y_augment_test_scaled_pred)
y_augment_test = y_scaler.inverse_transform(y_augment_test_scaled)

plt.figure(figsize=(10,7.5)), plt.plot(y_augment_test, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_augment_test_pred, 'r', linewidth=0.8, label='Predictions')
plt.title('Test data'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(6,6)), plt.plot(y_augment_test, y_augment_test_pred, '.', linewidth=0.8)
plt.title('Test data'), plt.ylabel('y(k): Measured'), plt.xlabel('y(k): Predicted')
plt.grid(which='both', axis='y', linestyle='--')

#%% infinite-step ahead predictions [first u_lags samples are used as initial conditions]
y_augment_test_scaled_sim = np.copy(y_test_scaled)

for sample in range(u_lags,len(u_test),1):
    regressorVector = np.hstack((u_test_scaled[sample-u_lags:sample,0], y_augment_test_scaled_sim[sample-y_lags:sample,0]))
    regressorVector = regressorVector[None,:]
    sim_response = model.predict(regressorVector) 
    y_augment_test_scaled_sim[sample] = sim_response

y_augment_test_sim =  y_scaler.inverse_transform(y_augment_test_scaled_sim)

#%% plot
plt.figure(figsize=(10,7.5)), plt.plot(y_test, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_augment_test_sim, 'r', linewidth=0.8, label='Predictions')
plt.title('Test data'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(6,6)), plt.plot(y_test, y_augment_test_sim, '.', linewidth=0.8)
plt.title('Test data'), plt.ylabel('y(k): Measured'), plt.xlabel('y(k): Predicted')
plt.grid(which='both', axis='y', linestyle='--')
    
    
    