# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:20:41 2017

@author: Christophe
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

dt = 0.001     # Time intervals that we will be looking at
T = 2          # Total simulation time
r = 5          # Firing rate in spikes per second
p = r*dt       # Probability of a spike in a very small time interval

n_trials = 500 # Total number of trials that we will simulate


# Here we run multiple simulations trials for a Poisson spiking neuron
# For each simulation trial we count the total number of spikes
# and the time between each spike to generate an inter-spike-interval
isi = np.zeros(0)
spike_count = np.zeros(n_trials)
for t in range(n_trials):
    spikes = np.random.rand(int(T/dt))
    spikes[spikes < p] = 1
    spikes[spikes < 1] = 0
    spike_count[t] = np.sum(spikes)
    isi = np.append(isi,np.diff(np.where(spikes==1)[0]))

# Plot the spike count histogram across trials
print(np.mean(spike_count))
print(np.var(spike_count))
bins = np.arange(np.min(spike_count),np.max(spike_count),1)
plt.hist(spike_count,bins-0.5,normed = True) # We subtract -0.5 because the values in bins represent the edges
plt.plot(bins,poisson.pmf(bins,mu=r*T),':k',linewidth=2.5)
plt.title('Spike count histogram')
plt.xlabel('Spike count')
plt.legend(['Poisson','Data'])
plt.show()

# Plot the inter spike interval distribution
#isi = isi*dt
bins = np.arange(np.min(isi),np.max(isi),10)
plt.hist(isi,bins,normed = True, histtype='barstacked')
plt.plot(bins,r*dt*np.exp(-r*bins*dt),':k',linewidth=3.0)
plt.title('ISI histogram')
plt.xlabel('ISI interval (ms)')
plt.legend(['Exponential','Data'])