##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                  Simulation analysis illustration

# Actual system: y(k)=0.3*y(k-1)+0.8*u(k-1)+0.2*u(k-1)^2+e(k)
# Model: y(k)=0.8*y(k-1)+0.6*u(k-1)+e(k)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import numpy as np, matplotlib.pyplot as plt

# package settings
np.random.seed(0)
plt.rcParams.update({'font.size': 14})

#%% define input signal (for illustration, let this be white noise)
N = 100
u = np.random.normal(0, 5, N)

plt.figure(figsize=(6,2.5))
plt.plot(u, 'steelblue', linewidth=0.8)
plt.ylabel('u'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

#%% compute actual process output
y = np.zeros((N, ))
e = np.random.normal(0, 0.1, N) # measurement noise
 
for k in range(1, N):
    y[k] = 0.3*y[k-1] + 0.8*u[k-1] + 0.2*u[k-1]*u[k-1] + e[k]

# plot
plt.figure(figsize=(6,2.5))
plt.plot(y, 'g', linewidth=0.8)
plt.ylabel('y'), plt.xlabel('seconds'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

#%% compute model's simulated output and compare
y_sim = np.zeros((N, ))

for k in range(1, N):
    y_sim[k] = 0.8*y_sim[k-1] + 0.6*u[k-1]

# plot
plt.figure(figsize=(6,2.5))
plt.plot(y, 'g', linewidth=0.8, label='Actual output')
plt.plot(y_sim, 'r', linewidth=0.8, label='Simulation response')
plt.ylabel('y simulated'), plt.xlabel('k'), plt.xlim(0), plt.legend()
plt.grid(which='both', axis='y', linestyle='--')

#%% compute model's 1-step ahead predictions
y_pred = np.zeros((N, ))

for k in range(1, N):
    y_pred[k] = 0.8*y[k-1] + 0.6*u[k-1]

# plot
plt.figure(figsize=(6,2.5))
plt.plot(y, 'g', linewidth=0.8, label='Actual output')
plt.plot(y_pred, 'r', linewidth=0.8, label='1-step ahead predictions')
plt.ylabel('y predictions'), plt.xlabel('k'), plt.xlim(0), plt.legend()
plt.grid(which='both', axis='y', linestyle='--')