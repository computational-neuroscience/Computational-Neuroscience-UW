# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:00:18 2016

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

# Plots the contribution of a particular state probability
# to the overall entropy of the distribution
p = np.arange(0.01,1,0.001)
i = -p*np.log2(p)
plt.plot(p,i)
plt.xlabel('probability')
plt.ylabel('Entropy value')

#%% Calculation of the car park example.
# a is an array which counts the number of occurences of the car being in that
# particular location
a = np.array([1, 1, 1, 1, 1, 1, 1, 1])
a = a/float(np.sum(a))
H = 0
for i in a:
    H += i*np.log2(i)
print str(-H)
#%% Entropy of a Bernoulli process with different values for p
p = np.arange(0.01,1,0.01)

y = []
for i in p:
    H = i*np.log2(i) + (1-i)*np.log2(1-i)
    y.append(-H)
    
plt.plot(p,y)
plt.xlabel('p')
plt.xlabel('H')
plt.title('Bernoulli distribution entropy as  a function of p')

#%% Mutual information
# Assume a response probability with maximum entropy

p_rp = 0.5 # Probability of a response
s_rp = 0.5 # Probability of a stimulus

HR = -(p_rp*np.log2(p_rp) + (1-p_rp)*np.log2(1-p_rp))

error_probabilities = np.arange(0.001,.501,0.001)
MI = np.zeros(len(error_probabilities))
for i in range(len(error_probabilities)):
    q = error_probabilities[i]
    HR_splus = -(q*np.log2(q) + (1-q)*np.log2(1-q))
    HR_smin  = -(q*np.log2(q) + (1-q)*np.log2(1-q))
    MI[i] = HR - (s_rp*HR_splus + (1-s_rp)*HR_smin)

plt.plot(error_probabilities,MI)
plt.title('Mutual information')
plt.xlabel('Error probability')
plt.ylabel('MI')
#%% Grandma's recipe for calculating mutual information with Poisson spiking neuron

# Get firing rate for Gaussian tuned cell
# x      : stimulus value
# x_0    :  prefered stimulus value
# s      : distribution standard deviation
# r_max  : maximal firing rate
# r_base : baseline firing rate
def gauss_tuning(x, x_0,s,r_max, r_base= 10):
    return r_base + r_max*np.exp(-0.5*((x-x_0)/s)**2)

#plt.plot(gauss_tuning(np.arange(0,2*np.pi,0.01),np.pi/2,5,10))
# Generates a spike train for a Poisson neuron firing with rate r_0 during an interval T
DELTA = 0.001 # time step for the simulations
def generate_poisson_spiketrain(r_0, T = 1):
    P = r_0*DELTA
    time_points = int(T/DELTA)
    
    spikes = np.random.rand(time_points)
    spikes[spikes < P] = 1
    spikes[spikes < 1] = 0
    return spikes

# Calculate the entropy of a frequency distribution of spike counts
def calculate_entropy(spike_counts):
    H = 0.0
    for s in np.unique(spike_counts):
        p = len(spike_counts[spike_counts == s])/float(len(spike_counts))
        H += -(p*np.log2(p))
    return H

# Parameters for the simulation
sigma_values = np.arange(0.1,8.0,0.1)
n_sigma = len(sigma_values)
MI = np.zeros(n_sigma)
for sigma_idx in range(n_sigma):
    sigma    = sigma_values[sigma_idx]       # standard deviation of the orientation tuning curve
    or_pref  = 0.0      # prefered orientation of the neuron
    r_base   = 10           # baseline firing rate for the neuron 
    n_trials = 500          # Number of stimulus presentations
    n_stim   = 20            # Number of different presented orientations
    trial_duration = 1.0    # Duration of a stimulus presentation in ms
    n_samples = int(trial_duration/DELTA)
    stim_range = np.linspace(0.0,2*np.pi,n_stim)
    
    # 1. Take a stimulus and present it many times and calculate the noise
    #    entropy for that stimulus. Do this for different stimulus values
    spike_counts = np.zeros((n_trials,n_stim))
    
    for s in range(n_stim):
        for t in range(n_trials):
            r = gauss_tuning(stim_range[s],or_pref,sigma,10)
            stim_st = generate_poisson_spiketrain(r,1)    
            spike_counts[t,s] = np.sum(stim_st)    
    
    # 2. Compute the variability due to noise
    NE = 0.0
    for i in range(n_stim):
        HR_S = calculate_entropy(spike_counts[:,i])
        NE += (1.0/n_stim)*HR_S
        
    
    # 4. Compute the total response entropy
    HR = calculate_entropy(spike_counts.flatten())
    MI[sigma_idx] = HR - NE
        
plt.plot(sigma_values,MI)
plt.title('Optimal standard deviation')
plt.xlabel('Standard deviation')
plt.ylabel('MI')
#%% Produces histogram for the spike count frequencies
plt.clf()
plt.subplot(1,2,1)
plt.hist(spike_counts.flatten())
plt.title('Response distribution')
plt.xlabel('Spikes counted')
plt.ylabel('Frequency')
plt.subplot(1,2,2)
for s in range(n_stim):
    plt.hist(spike_counts[:,s],alpha = 0.5)
plt.title('Conditional response distributions')
plt.xlabel('Spikes counted')
plt.ylabel('Frequency')