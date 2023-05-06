##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                      LSTM-NARX-based Modeling of Heat Exchangers 
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

#%% split into fitting and validation dataset
u_fit = u[:3000,0:1]; u_val = u[3000:,0:1] 
y_fit = y[:3000,0:1]; y_val = y[3000:,0:1] 

#%% scale data before model fitting
u_scaler = StandardScaler(); u_fit_scaled = u_scaler.fit_transform(u_fit); u_val_scaled = u_scaler.transform(u_val) 
y_scaler = StandardScaler(); y_fit_scaled = y_scaler.fit_transform(y_fit); y_val_scaled = y_scaler.transform(y_val) 

X_fit_scaled = np.hstack((u_fit_scaled, y_fit_scaled)); X_val_scaled = np.hstack((u_val_scaled, y_val_scaled))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          re-arrage data with time steps
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rearrange X data into (# sequence samples, # time steps, # features) form
nTimeSteps = 9
X_fit_scaled_sequence = []; X_val_scaled_sequence = []
y_fit_scaled_sequence = []; y_val_scaled_sequence = []

for sample in range(nTimeSteps, X_fit_scaled.shape[0]):
    X_fit_scaled_sequence.append(X_fit_scaled[sample-nTimeSteps:sample,:])
    y_fit_scaled_sequence.append(y_fit_scaled[sample])
    
for sample in range(nTimeSteps, X_val_scaled.shape[0]):
    X_val_scaled_sequence.append(X_val_scaled[sample-nTimeSteps:sample,:])
    y_val_scaled_sequence.append(y_val_scaled[sample])

# X conversion: convert list of (time steps, features) arrays into (samples, time steps, features) array 
X_fit_scaled_sequence, y_fit_scaled_sequence = np.array(X_fit_scaled_sequence), np.array(y_fit_scaled_sequence) 
X_val_scaled_sequence, y_val_scaled_sequence = np.array(X_val_scaled_sequence), np.array(y_val_scaled_sequence) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          fit RNN model
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# import Keras libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import regularizers

# define model
model = Sequential()
model.add(LSTM(units=6, kernel_regularizer=regularizers.L1(0.001), input_shape=(nTimeSteps,2)))
model.add(Dense(units=1))

#%% model summary
model.summary()

#%% compile model
model.compile(loss='mse', optimizer='Adam')

#%% fit model with early stopping
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_fit_scaled_sequence, y_fit_scaled_sequence, epochs=250, batch_size=125, validation_data=(X_val_scaled_sequence, y_val_scaled_sequence), callbacks=[es])

#%% plot validation curve
plt.figure()
plt.title('Validation Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.grid()
plt.show()

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         Residual analysis on fitting dataset
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% get model predictions (1-step ahead) and residuals on training dataset
y_fit_scaled_sequence_pred = model.predict(X_fit_scaled_sequence) 
y_fit_sequence_pred =  y_scaler.inverse_transform(y_fit_scaled_sequence_pred)
y_fit_sequence = y_scaler.inverse_transform(y_fit_scaled_sequence)
residuals_fit = y_fit_sequence - y_fit_sequence_pred

plt.figure(figsize=(10,3.5)), plt.plot(y_fit_sequence, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_fit_sequence_pred, 'r', linewidth=0.8, label='Predictions')
plt.title('Fitting data'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(6,2.5)), plt.plot(residuals_fit, 'black', linewidth=0.8)
plt.title('Fitting data'), plt.ylabel('residuals'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

#%% ACF of residuals
conf_int = 1.96/np.sqrt(len(residuals_fit))

plot_acf(residuals_fit, lags= 40, alpha=None, title='')
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag'), plt.title('Autocorrelations'), plt.ylim(-0.2, 0.5) 
plt.show()

#%% CCF b/w residuals and input sequence
ccf_vals = ccf(residuals_fit, u_fit[nTimeSteps:], adjusted=False) # ccf for lag > 0
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
##               m-step ahead predictions on validation dataset
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% 1-step ahead predictions
y_val_scaled_sequence_pred = model.predict(X_val_scaled_sequence) 
y_val_sequence_pred =  y_scaler.inverse_transform(y_val_scaled_sequence_pred)
y_val_sequence = y_scaler.inverse_transform(y_val_scaled_sequence)

plt.figure(figsize=(10,7.5)), plt.plot(y_val_sequence, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_val_sequence_pred, 'r', linewidth=0.8, label='Predictions')
plt.title('Validation data'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(6,6)), plt.plot(y_val_sequence, y_val_sequence_pred, '.', linewidth=0.8)
plt.title('Validation data'), plt.ylabel('y(k): Measured'), plt.xlabel('y(k): Predicted')
plt.grid(which='both', axis='y', linestyle='--')

#%% infinite-step ahead predictions [first nTimeSteps samples are used as initial conditions]
y_val_scaled_sim = np.copy(y_val_scaled)

for sample in range(nTimeSteps, X_val_scaled.shape[0]):
    X_val_scaled_sim = np.hstack((u_val_scaled,y_val_scaled_sim))
    inputSequence = X_val_scaled_sim[sample-nTimeSteps:sample,:]
    inputSequence = inputSequence[None,:,:]
    sim_response = model.predict(inputSequence) 
    y_val_scaled_sim[sample] = sim_response

y_val_pred_sim =  y_scaler.inverse_transform(y_val_scaled_sim)

#%% plot
plt.figure(figsize=(10,7.5)), plt.plot(y_val, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_val_pred_sim, 'r', linewidth=0.8, label='Predictions')
plt.title('Validation data'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(6,6)), plt.plot(y_val, y_val_pred_sim, '.', linewidth=0.8)
plt.title('Validation data'), plt.ylabel('y(k): Measured'), plt.xlabel('y(k): Predicted')
plt.grid(which='both', axis='y', linestyle='--')
    
    
    