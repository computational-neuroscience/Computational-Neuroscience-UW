# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:52:52 2016

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Modeling the postsynaptic conductance
# P_s is a numerical simulation of the kinematic model ofor a synapse
# P_s_e is an analytical approximation
dt  = 0.01 # [ms]
a_s = 0.93 # [ms]
b_s = 0.19 # [ms]

n_steps = 3000
P_s  = np.zeros(n_steps)
time = dt*np.arange(0,n_steps)
for t in range(1,n_steps):
    if t*dt > 1:
        a_s = 0
    
    dP_s = a_s*(1-P_s[t-1]) - b_s*P_s[t-1]
    P_s[t] = P_s[t-1] + dP_s*dt

P_max = np.max(P_s)

P_s_e = P_max*np.exp(-time*b_s)

plt.plot(time,P_s)
plt.plot(time,P_s_e)
plt.xlabel('Time (ms)')
plt.ylabel(r'$P_S$')
plt.title('Postsynaptic channel conductance')
plt.legend(['Kinematic model','Filter model'])
#%% Modeling arrival of multiple spikes
r  = 0.02 # Spikes per millisecond of presynaptic neuron
dt = 0.1  # simulation step [ms]
tau_s = 1/b_s

n_steps = 5000 # simulates 0.5 seconds
P_s  = np.zeros(n_steps)
time = dt*np.arange(0,n_steps)
for t in range(1,n_steps):
    if r*dt > np.random.rand():
        P_s[t] = P_s[t-1] + P_max*(1-P_s[t-1])
    else:
        P_s[t] = P_s[t-1] - (P_s[t-1]/tau_s)*dt

plt.plot(time,P_s)
    
#%% Modeling GABA_A and NMDA as a difference of two gaussian

