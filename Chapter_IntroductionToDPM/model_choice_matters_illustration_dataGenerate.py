##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                      Why careful choice of model matters
##               Generate data from true process with SNR ~ 10
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from sippy import functionset as fset

#%% generate white-noise-like input
np.random.seed(10)

prob_switch_1 = 0.05
[u,_,_] = fset.GBN_seq(1000, prob_switch_1)  

# plot u
plt.figure(figsize=(4,1.5))
plt.plot(u, 'steelblue', linewidth=0.8)
plt.ylabel('u'), plt.xlabel('k'), plt.xlim(0)

#%% generate internal state x
x = np.zeros((1000,))

for k in range(2,1000):
    x[k] = 0.8*x[k-1] + 0.5*u[k-1] # the internal process dynamics

# plot x
plt.figure(figsize=(4,1.5))
plt.plot(x, 'g', linewidth=0.8)
plt.ylabel('x'), plt.xlabel('k'), plt.xlim(0)

#%% generate output noise e
x_var = np.var(x)
SNR = 10
e_var = x_var/SNR
e_std = np.sqrt(e_var)

np.random.seed(10)
e = np.random.normal(loc=0, scale=e_std, size=(1000,))

# plot e
plt.figure(figsize=(4,1.5))
plt.plot(e, 'black', linewidth=0.8)
plt.ylabel('e'), plt.xlabel('k'), plt.xlim(0)

#%% generate output y
y = x + e

# plot y
plt.figure(figsize=(4,1.5))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y'), plt.xlabel('k'), plt.xlim(0)

#%% save data
simpleProcessData = np.column_stack((u, y))
np.savetxt('simpleProcess.csv', simpleProcessData, delimiter=',')