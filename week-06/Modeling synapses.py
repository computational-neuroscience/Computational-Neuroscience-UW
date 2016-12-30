# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:52:52 2016

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Illustration of probability of synaptic connection
tau_ampa   = 2  # [ms]
tau_gaba_a = 2 # [ms]
tau_nmda   = 10  # [ms]

t = np.arange(0,100,0.1)

ampa   = np.exp(-t/tau_ampa)
gaba_a = (t/tau_gaba_a)*np.exp(1-(t/tau_gaba_a))
nmda   = (t/tau_nmda)*np.exp(1-(t/tau_nmda))
plt.clf()
plt.plot(t,ampa)
plt.plot(t,gaba_a)
plt.plot(t,nmda)
plt.axvline(tau_gaba_a,color ='g')
#%% Linear filter model of a synapse
T  = 200 # simulation interval [ms]
dt = 1.0   # simulation timestep [ms]
spike_times = np.array([50, 80, 95, 150])

# Calculate the filter response for each spike
K = np.zeros((len(spike_times),int(T/dt)))
for i in range(len(spike_times)):
    t_i = np.arange(0,(T-spike_times[i]),dt)
    filter_response = np.exp(-t_i/6.0)
    K[i,int(spike_times[i]/dt):] = filter_response

# Step by step integration
alpha = np.zeros(int(T/dt))
for t in range(int(T/dt)):
    for spike in spike_times:
        if spike <= t*dt:
            alpha[t] += (1/6.0)*(t*dt - spike)*np.exp(1-(t*dt - spike)/6.0)
            
plt.clf()
plt.subplot(1,3,1)
plt.title('Individual filter response')
plt.plot(np.arange(0,T,dt),K.transpose(),color='k')
plt.xlabel('Time [ms]')
plt.ylabel('P')

plt.subplot(1,3,2)
plt.title('Linear summation of AMPA filters')
plt.plot(np.arange(0,T,dt),np.sum(K,0),'k')
plt.xlabel('Time [ms]')
plt.ylabel('P')

plt.subplot(1,3,3)
plt.title('Linear summation of GABA_A filters')
plt.plot(np.arange(0,T,dt),alpha,'k')
#%% Example with integrate and fire neurons
# Neuron parameters
E_S      = -80.0  # [mV]
E_L      = -70.0  # [mV]
V_thresh = -54.0  # [mV]
V_max    =  40.0  # [mV]
R        =  20.0  # [Mohm]
C        =   1.0  # [nF]
I        =   1.25 # [nA]
g_max    =   0.009
tau_p    =  4.9   # [ms]
tau_m    = R*C    # [ms]

T  = 500
dt = 0.1
t  = np.arange(0,T,dt)

# voltage variables and lists for spike times
V_0 = np.zeros(len(t))
V_1 = np.zeros(len(t))
V_0[0] = E_L
V_1[0] = E_L + (V_thresh-E_L)/4.0

alpha_0 = np.zeros(len(t))
alpha_1 = np.zeros(len(t))
S_0 = []
S_1 = []

refrac_0 = 0
refrac_1 = 0

for i in range(1,len(t)):
    # Calculate the current alpha values
    for spike in S_0:
        alpha_1[i] += (1/tau_p)*(t[i]-spike)*np.exp(1-(t[i]-spike)/tau_p)
    for spike in S_1:
        alpha_0[i] += (1/tau_p)*(t[i]-spike)*np.exp(1-(t[i]-spike)/tau_p)
        
    # Update the voltage
    dV_0 = - ( (V_0[i-1] - E_L) + g_max*alpha_0[i]*(V_0[i-1] - E_S)*R) + R*I
    dV_1 = - ( (V_1[i-1] - E_L) + g_max*alpha_1[i]*(V_1[i-1] - E_S)*R) + R*I
    
    if refrac_0 > 0:
        V_0[i] = E_L
        refrac_0 -= 1
    else:
        V_0[i] = V_0[i-1] + (dV_0/tau_m)*dt
    
    if refrac_1 > 0:
        V_1[i] = E_L
        refrac_1 -= 1
    else:
        V_1[i] = V_1[i-1] + (dV_1/tau_m)*dt

    # Check for spikes
    if V_0[i] > V_thresh:
        S_0.append(i*dt)
        V_0[i] = V_max
        refrac_0 = 1
    if V_1[i] > V_thresh:
        S_1.append(i*dt)
        V_1[i] = V_max
        refrac_1 = 1

# Plot the results
plt.clf()
plt.subplot(2,2,1)
plt.plot(t,V_0,'g')
plt.plot(t,V_1,'r')
plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')

plt.subplot(2,2,2)
plt.plot(t,alpha_0)
plt.plot(t,alpha_1)
plt.ylim(0,1.2)
plt.subplot(2,2,3)
plt.plot(t,V_0,'k')
#plt.plot(t,a,'--k')
plt.plot(t,30*alpha_0)

plt.subplot(2,2,4)
plt.plot(t,V_1,'k')
#plt.plot(t,b,'--k')
plt.plot(t,30*alpha_1)
# Measure of synchrony
min_len = min([len(S_0),len(S_1)])
spike_diff = np.array(S_0[:min_len]) - np.array(S_1[:min_len])
print (spike_diff)
print (np.diff(S_0))
print (np.diff(S_1))