##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##              Fit ARIMA model using ACF & PACF plots and ADF test
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams.update({'font.size': 14})
np.random.seed(100)

#%% read data
y = np.loadtxt('ARIMA_data.txt')

# plot
plt.figure(figsize=(5,2))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y(k)'), plt.xlabel('k'), plt.xlim(0)

#%% generate ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib.ticker import MaxNLocator
conf_int = 2/np.sqrt(len(y))

fig, ax = plt.subplots(1,1,figsize=(4,2.5))
plot_acf(y, lags= 20, alpha=None, title='', ax=ax)
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag'), plt.ylim(-0.25, 1.1)
plt.show()

fig, ax = plt.subplots(1,1,figsize=(4,2.5))
plot_pacf(y, lags= 20, alpha=None, title='', ax=ax)
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag'), plt.ylim(-0.25, 1.1)
plt.show()

#%% perform augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)
print('p-value: %f' % result[1])

#%% check ACF/PACF/ADF test of 1-degree differenced signal
delta_y = np.diff(y, axis=0)

# plot
plt.figure(figsize=(5,2))
plt.plot(delta_y, 'g', linewidth=0.8)
plt.ylabel('y(k)-y(k-1)'), plt.xlabel('k'), plt.xlim(0)

# ACF
fig, ax = plt.subplots(1,1,figsize=(4,2.5))
plot_acf(delta_y, lags= 20, alpha=None, ax=ax)
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag'), plt.ylim(-0.5, 1.1)
plt.show()

# PACF
fig, ax = plt.subplots(1,1,figsize=(4,2.5))
plot_pacf(delta_y, lags= 20, alpha=None, ax=ax)
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag'), plt.ylim(-0.5, 1.1)
plt.show()

# ADF test
from statsmodels.tsa.stattools import adfuller
result = adfuller(delta_y)
print('p-value: %f' % result[1])

#%% fit ARIMA model
y_centered = y - np.mean(y)
model = ARIMA(y_centered, order=(1, 1, 1)) # order = (p,d,r)
results = model.fit()

# Print out summary information on the fit
print(results.summary())
