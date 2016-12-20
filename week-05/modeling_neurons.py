# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:53:46 2016

@author: Christophe
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the passive membrane model
R = 3
C = 0.01
tau = R*C
V_rest = 0.0

t = np.arange(0,0.5,0.001)

# The exact solution has to be calculated during  three different intervals
I_ext = 0.0
V_init = 0.0
V_1 = (V_rest + R*I_ext)*(1-np.exp(-t/tau)) + V_init*np.exp(-t/tau)

I_ext = 0.5
V_init = V_1[-1]
V_2 = (V_rest + R*I_ext)*(1-np.exp(-t/tau)) + V_init*np.exp(-t/tau)

I_ext = 0.0
V_init = V_2[-1]
V_3 = (V_rest + R*I_ext)*(1-np.exp(-t/tau)) + V_init*np.exp(-t/tau)

V_exact = np.concatenate((V_1,V_2,V_3))
I = np.concatenate((np.zeros(len(t)),0.5*np.ones(len(t)),np.zeros(len(t))))

# Euler solution
V_euler = np.zeros(len(I))
for i in range(1,len(I)):
    dV = (-(V_euler[i-1] - V_rest) + R*I[i])/tau
    V_euler[i] = V_euler[i-1] + dV*0.001

plt.subplot(1,2,1)
plt.plot(np.arange(0,1.5,0.001),V_exact,'k')
plt.plot(np.arange(0,1.5,0.001),I,'r')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.title('Exact solution')
plt.subplot(1,2,2)
plt.plot(np.arange(0,1.5,0.001),V_euler,'--k')
plt.plot(np.arange(0,1.5,0.001),I,'r')
plt.title('Euler approximation')
#%%