# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 16:59:15 2016

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

DELTA = 0.0005 # time step for the simulations

# Generates a spike train for a Poisson neuron firing with rate r_0 during an interval T
def generate_poisson_spiketrain(r_0, T = 1):
    P = r_0*DELTA
    time_points = int(T/DELTA)
    
    spikes = np.random.rand(time_points)
    spikes[spikes < P] = 1
    spikes[spikes < 1] = 0
    return spikes

# Generates a spike train for a Poisson neuron firing with rate r_0 during an interval T
# After a spike has been emitted, the firing rate is temporarily reset to zero and 
# climbs back to the actual rate with time constant tau_r
def generate_refractory_spiketrain(r_0, tau_r, T=1):
    time_points = int(T/DELTA)
    
    spikes = np.zeros(time_points)
    r = r_0
    for i in range(l):
        P = r*DELTA
        s = np.random.rand(1)
        
        r += ((r_0-r)/tau_r)*DELTA
        if s < P:
            spikes[i] = 1
            r = 0
    return spikes

#%% Rasterplot for different spike trains
n_spike_trains =100
T = 5
l = int(T/DELTA)
spikes = np.zeros((n_spike_trains,l))
for idx in range(n_spike_trains):
    spikes[idx,:] = generate_refractory_spiketrain(10,0.01,T)

plt.clf()
plt.spy(spikes,aspect='auto')
plt.title('Raster plot')
plt.xlabel('Time')
plt.ylabel('Instance')
#%% Exercise 1a
# Distribution of Poisson spiking neuron
r = 100 # firing rate of the neuron (Hz)
T = 1   # Duration of a single simulation (s)

n_simulations = 100
n_spikes = np.zeros(n_simulations)
for idx in range(n_spike_trains):    
    n_spikes[idx] = sum(generate_poisson_spiketrain(r,T))
   
plt.hist(n_spikes)
M = np.mean(n_spikes)
VAR = np.var(n_spikes)
print M,VAR
#%% Exercise 1b
# Distribution of interspike intervals
r = 60
T = 100
spikes = generate_poisson_spiketrain(r,T)
spike_indices = spikes.nonzero()[0]
isi = np.diff(spike_indices)*DELTA
M = np.mean(isi)
V = np.var(isi)
CV = np.sqrt(V)/M

# Fano factor when counting spikes using different bin sizes
binSizes = np.arange(0.001,0.101,0.001) # in seconds
F = np.zeros(len(binSizes))
for i in range(len(binSizes)):
    bins = np.arange(0,T/DELTA,binSizes[i]/DELTA)
    y = np.histogram(spike_indices,bins)[0]
    
    M = np.mean(y)
    V = np.var(y)
    F[i] = V/M
    

plt.subplot(121)
plt.hist(isi,30)
plt.title("CV: " + str(CV))
plt.subplot(122)
plt.plot(binSizes,F)
plt.hlines(1,binSizes[0],binSizes[-1])
#%%
tau_r = np.arange(0.001,0.021,0.001)
r_0 = 30
T = 10

CV = np.zeros(len(tau_r))
for i in range(len(tau_r)):
    spikes = generate_refractory_spiketrain(r_0, tau_r[i],T)
    #spikes = generate_poisson_spiketrain(r_0,T)
    spike_indices = spikes.nonzero()[0]
    isi = np.diff(spike_indices)*DELTA
    CV[i] = np.sqrt(np.var(isi))/np.mean(isi)
    
plt.plot(tau_r,CV)

    
    