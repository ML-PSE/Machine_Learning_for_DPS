##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                      ADF test for non-stationarity check
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

plt.rcParams.update({'font.size': 14})  
np.random.seed(10)

#%% generate data
ar_coeffs = np.array([1, -0.96]) # y(k) = 0.96y(k-1) + e(k) 
ARprocess = ArmaProcess(ar=ar_coeffs)
y = ARprocess.generate_sample(nsample=1000)

# plot
plt.figure(figsize=(4,2))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y(k)'), plt.xlabel('k'), plt.xlim(0)

#%% perform augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)
print('p-value: %f' % result[1])