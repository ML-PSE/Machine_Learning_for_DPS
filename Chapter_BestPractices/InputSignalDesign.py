##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                             Persistant input signals
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
from sippy import functionset as fset
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

#%% generate GBN input signal using SIPPY and plot
[gbn_signal, _, _] = fset.GBN_seq(1000, 0.05)

# plot
plt.figure(figsize=(6,1.5))
plt.plot(gbn_signal, 'steelblue', linewidth=0.8)
plt.ylabel('GBN_signal'), plt.xlabel('k'), plt.xlim(0)

#%% generate PRBS input signal and plot
from scipy.signal import max_len_seq

PRBS_signal = max_len_seq(6, length=100)[0]*2-1  # +1 and -1

# plotCreate a new figure.
plt.figure(figsize=(14,3.5))
plt.plot(PRBS_signal, 'steelblue', linewidth=1.8, drawstyle='steps')
plt.ylabel('PRBS_signal'), plt.xlabel('k'), plt.xlim(0)
