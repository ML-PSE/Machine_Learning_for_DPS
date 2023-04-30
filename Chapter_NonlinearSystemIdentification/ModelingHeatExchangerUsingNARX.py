##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                      Polynimial NARX-based Modeling of Heat Exchangers 
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import ccf
from matplotlib.ticker import MaxNLocator

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.display_results import results

# package settings
plt.rcParams.update({'font.size': 14})

#%% read data and plot
data = np.loadtxt('exchanger.dat')
u = data[:,1, None]; y = data[:,2, None]

# plots
plt.figure(figsize=(10,4.5))
plt.plot(u, 'steelblue', linewidth=0.8, drawstyle='steps')
plt.ylabel('u(k)'), plt.xlabel('k'), plt.xlim(0)

plt.figure(figsize=(10,4.5))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y(k)'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

#%% split into training and test dataset
u_fit = u[:3000,0:1]; u_test = u[3000:,0:1] 
y_fit = y[:3000,0:1]; y_test = y[3000:,0:1] 

#%% center data before model fitting
u_scaler = StandardScaler(with_std=False); u_fit_centered = u_scaler.fit_transform(u_fit); u_test_centered = u_scaler.transform(u_test) 
y_scaler = StandardScaler(with_std=False); y_fit_centered = y_scaler.fit_transform(y_fit); y_test_centered = y_scaler.transform(y_test) 

#%% fit NARX model
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=75,
    ylag=3, xlag=9,
    info_criteria='aic',
    estimator='least_squares',
    basis_function=basis_function
)

model.fit(X=u_fit_centered, y=y_fit_centered)

#%% check AIC values
xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel('n_terms')
plt.ylabel('Information Criteria')

#%% see the regressors
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                         Residual analysis on fitting dataset
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% get model predictions and residuals on training dataset
y_fit_predicted_centered = model.predict(X=u_fit_centered, y=y_fit_centered, steps_ahead=1)
y_fit_predicted = y_scaler.inverse_transform(y_fit_predicted_centered)
residuals_fit = y_fit - y_fit_predicted

plt.figure(figsize=(10,3.5)), plt.plot(y_fit, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_fit_predicted, 'r', linewidth=0.8, label='Predictions')
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
ccf_vals = ccf(residuals_fit, u_fit, adjusted=False) # ccf for lag > 0
ccf_vals = ccf_vals[1:41] # ccf for lag 1 to 20

# generate CCF plot
lags = np.arange(1,41)

plt.figure(figsize=(6,4)), plt.vlines(lags, [0], ccf_vals), plt.axhline(0, 0, lags[-1])
plt.plot(lags, ccf_vals, marker='o', markersize=5, linestyle='None')
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.xlabel('lag'), plt.title('Cross-correlations')
plt.ylim(-0.5, 0.5) 

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##               m-step ahead predictions on test dataset
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% 1-step ahead predictions
y_test_predicted_centered = model.predict(X=u_test_centered, y=y_test_centered, steps_ahead=1)
y_test_predicted = y_scaler.inverse_transform(y_test_predicted_centered)

plt.figure(figsize=(15,7.5)), plt.plot(y_test, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_test_predicted, 'r', linewidth=0.8, label='Predictions')
plt.title('Test data'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(7,7)), plt.plot(y_test, y_test_predicted, '.', linewidth=0.8)
plt.title('Test data (1-step ahead predictions)'), plt.ylabel('y(k): Measured'), plt.xlabel('y(k): Predicted')
plt.grid(which='both', axis='y', linestyle='--')

#%% 10-step ahead predictions
y_test_predicted_centered = model.predict(X=u_test_centered, y=y_test_centered, steps_ahead=10)
y_test_predicted = y_scaler.inverse_transform(y_test_predicted_centered)

plt.figure(figsize=(15,7.5)), plt.plot(y_test, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_test_predicted, 'r', linewidth=0.8, label='Predictions')
plt.title('Test data'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(7,7)), plt.plot(y_test, y_test_predicted, '.', linewidth=0.8)
plt.title('Test data (10-step ahead predictions)'), plt.ylabel('y(k): Measured'), plt.xlabel('y(k): Predicted')
plt.grid(which='both', axis='y', linestyle='--')

#%% infinite-step ahead predictions
y_test_predicted_centered = model.predict(X=u_test_centered, y=y_test_centered, steps_ahead=None)
y_test_predicted = y_scaler.inverse_transform(y_test_predicted_centered)

plt.figure(figsize=(10,7.5)), plt.plot(y_test, 'g', linewidth=0.8, label='Measurements'), plt.plot(y_test_predicted, 'r', linewidth=0.8, label='Predictions')
plt.title('Test data (simulation response)'), plt.ylabel('y(k): Measured vs predicted'), plt.xlabel('k'), plt.legend(), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(6,6)), plt.plot(y_test, y_test_predicted, '.', linewidth=0.8)
plt.title('Test data (simulation response)'), plt.ylabel('y(k): Measured'), plt.xlabel('y(k): Predicted')
plt.grid(which='both', axis='y', linestyle='--')