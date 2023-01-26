##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                        PSD Illustration
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

#%% read data
data = np.loadtxt('simpleInputOutput.csv', delimiter=',')
u = data[:,0]; y = data[:,1]

# time-plot
# plot y
plt.figure(figsize=(6,1.5))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y'), plt.xlabel('k'), plt.xlim(0)

# plot u
plt.figure(figsize=(6,1.5))
plt.plot(u, 'steelblue', linewidth=0.8)
plt.ylabel('u'), plt.xlabel('k'), plt.xlim(0)

#%% Periodograms
# input PSD
freq, PSD = signal.welch(u)

plt.figure(figsize=(5,3)), plt.plot(freq, PSD, 'darkviolet', linewidth=0.8)
plt.ylabel('input PSD'), plt.xlabel('frequency [Hz]'), plt.xlim(0)

# output PSD
freq, PSD  = signal.welch(y)

plt.figure(figsize=(5,3)), plt.plot(freq, PSD, 'darkviolet', linewidth=0.8)
plt.ylabel('output PSD'), plt.xlabel('frequency [Hz]'), plt.xlim(0)

