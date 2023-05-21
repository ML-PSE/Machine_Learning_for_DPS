##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                      CVA modeling of glass furnaces
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import matplotlib.pyplot as plt, numpy as np
from sklearn.preprocessing import StandardScaler
from sippy import system_identification as SysID

# package settings
plt.rcParams.update({'font.size': 20})

#%% read data and plot
data = np.loadtxt('glassfurnace.dat'); U = data[:,1:4]; Y = data[:,4:]

# plots
plt.figure(figsize=(10,3))
plt.plot(U[:,0], 'steelblue', linewidth=0.8, drawstyle='steps')
plt.ylabel('u1(k)'), plt.xlabel('k'), plt.xlim(0)

plt.figure(figsize=(10,3))
plt.plot(U[:,1], 'steelblue', linewidth=0.8, drawstyle='steps')
plt.ylabel('u2(k)'), plt.xlabel('k'), plt.xlim(0)

plt.figure(figsize=(10,3))
plt.plot(U[:,2], 'steelblue', linewidth=0.8, drawstyle='steps')
plt.ylabel('u3(k)'), plt.xlabel('k'), plt.xlim(0)

plt.figure(figsize=(10,3))
plt.plot(Y[:,:2], linewidth=0.8)
plt.ylabel('y1(k) and y2(k)'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(10,3))
plt.plot(Y[:,2:4], linewidth=0.8)
plt.ylabel('y3(k) and y4(k)'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

plt.figure(figsize=(10,3))
plt.plot(Y[:,4:], linewidth=0.8)
plt.ylabel('y5(k) and y6(k)'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

#%% split into training and test dataset
from sklearn.model_selection import train_test_split
U_train, U_test, Y_train, Y_test = train_test_split(U, Y, test_size=0.2, shuffle=False)

plt.figure(figsize=(10,3))
plt.plot(U_test[:,2], 'steelblue', linewidth=0.8, drawstyle='steps')
plt.ylabel('u3(k)'), plt.xlabel('k'), plt.xlim(0)

plt.figure(figsize=(10,3))
plt.plot(Y_test[:,0], 'g', linewidth=0.8)
plt.ylabel('y1(k)'), plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--')

#%% center data before model fitting
U_scaler = StandardScaler(with_std=False); U_train_centered = U_scaler.fit_transform(U_train); U_test_centered = U_scaler.transform(U_test) 
Y_scaler = StandardScaler(with_std=False); Y_train_centered = Y_scaler.fit_transform(Y_train); Y_test_centered = Y_scaler.transform(Y_test) 

#%% fit CVA model
model = SysID(Y_train_centered, U_train_centered, 'CVA', IC='AIC', SS_f=20, SS_orders=[1,20])
print(model.G)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                 compare simulation responses vs measurements
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% compare simulation responses vs measurements on training data
from sippy import functionsetSIM as fsetSIM

Xid_train, Yid_train_centered = fsetSIM.SS_lsim_process_form(model.A, model.B, model.C, model.D, np.transpose(U_train_centered), model.x0)
Yid_train_centered = np.transpose(Yid_train_centered)

#%% plots
i = 5
plt.figure(figsize=(10,3))
plt.plot(Y_train_centered[:,i], 'g', linewidth=0.8)
plt.plot(Yid_train_centered[:,i], 'r', linewidth=0.8)
plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--'), plt.title('y%s (k)' %str(i))

#%% compare simulation responses vs measurements on test data
Xid_test, Yid_test_centered = fsetSIM.SS_lsim_process_form(model.A, model.B, model.C, model.D, np.transpose(U_test_centered), Xid_train[:,-1, None])
Yid_test_centered = np.transpose(Yid_test_centered)

#%% plot
i = 5
plt.figure(figsize=(10,3))
plt.plot(Y_test_centered[:,i], 'g', linewidth=0.8)
plt.plot(Yid_test_centered[:,i], 'r', linewidth=0.8)
plt.xlabel('k'), plt.xlim(0)
plt.grid(which='both', axis='y', linestyle='--'), plt.title('y%s (k)' %str(i))