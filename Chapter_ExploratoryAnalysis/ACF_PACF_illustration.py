##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                              ACF and PACF Illustration
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from matplotlib.ticker import MaxNLocator

#%% read data
y = np.loadtxt('simpleTimeSeries.csv', delimiter=',')

# time-plot
plt.figure(figsize=(6,3))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y'), plt.xlabel('k'), plt.xlim(0)

#%% generate ACF plot
conf_int = 2/np.sqrt(len(y))

plot_acf(y, lags= 20, alpha=None) # alpha=None avoids plot_acf's inbuilt confidence interval plotting
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag')
plt.show()

#%% generate PACF plot
conf_int = 2/np.sqrt(len(y))

plot_pacf(y, lags= 20, alpha=None) # alpha=None avoids plot_acf's inbuilt confidence interval plotting
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.xlabel('lag')
plt.show()