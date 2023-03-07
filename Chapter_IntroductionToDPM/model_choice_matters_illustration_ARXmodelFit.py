##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                      Why careful choice of model matters
##                            Fit ARX model via OLS
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import numpy as np
import statsmodels.api as sm # statsmodels provides parameter error estimates as well

#%% read data
simpleProcessData = np.loadtxt('simpleProcess.csv', delimiter=',')
u = simpleProcessData[:,0]
y = simpleProcessData[:,1]

#%% generate response and regressors for ARX model
y_centered = y - np.mean(y)
u_centered = u - np.mean(u)

y_arx = y_centered[1:]
regressor_arx = np.column_stack((y_centered[:-1], u_centered[:-1]))

#%% fit ARX model via OLS
model = sm.OLS(y_arx, regressor_arx)
results = model.fit()
print(results.summary())