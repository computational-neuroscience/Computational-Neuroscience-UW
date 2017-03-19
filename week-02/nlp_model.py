# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:26:24 2017

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Time reversal filter described by Adelson and Bergen (1985)
L = 300             # range [ms]
alpha = 1.0/15      # [ms]
t = np.arange(0,L,1)

D = alpha*np.exp(-alpha*t)*( ((alpha*t)**5)/(np.math.factorial(5)) - ((alpha*t)**7)/np.math.factorial(7))
plt.plot(-t,D)

#%% Linear stage: filtering of the response
n_samples   = 500000
noise_stim  = np.random.normal(0.0,1.0,n_samples)
linear_resp = np.convolve(noise_stim,D)[:n_samples]
plt.subplot(1,2,1)
plt.plot(noise_stim)
plt.subplot(1,2,2)
plt.plot(linear_resp)

#%% Nonlinear stage: baseline firing rate r_0 followed by a linear increase
# until r_max
r_0 = 0    # Baseline firing rate [Hz]
r_max = 70 # maximam firing rate

F_L = linear_resp.copy()
F_L[F_L < 0] = 0
F_L = r_max*(F_L/np.max(F_L))
r_est = r_0 + F_L
plt.plot(r_est)

#%% Poisson spike generation
p = r_est*0.001
spikes = np.random.rand(n_samples)
spikes[spikes < p] = 1
spikes[spikes < 1] = 0

#%% Calculate the spike triggered average stimulus
spike_times = np.where(spikes == 1)[0]
sta = np.zeros(L)
for i in spike_times:
    if i < L:
        continue
    
    sta += noise_stim[(i-L):i]

sta = sta/np.sum(spike_times > L)
plt.plot(sta)
#%% Estimating the nonlinear part using histogram technique
bins = np.arange(np.min(linear_resp),np.max(linear_resp),0.01)

raw_hist = np.histogram(linear_resp,bins)[0]
sta_hist = np.histogram(linear_resp[spikes == 1],bins)[0]

nl_estimate = np.true_divide(sta_hist,raw_hist)

plt.subplot(1,2,1)
plt.hist(linear_resp,bins, normed = True,alpha = 0.5)
plt.hist(linear_resp[spikes == 1],bins, normed = True,alpha = 0.5)
plt.subplot(1,2,2)
plt.plot(bins[:-1],nl_estimate)

#%% Demonstration of a nonlinear pooling cell
D1 = alpha*np.exp(-alpha*t)*( ((alpha*t)**5)/(np.math.factorial(5)))
D2 = -alpha*np.exp(-alpha*t)*( ((alpha*t)**5)/(np.math.factorial(5)))
plt.plot(-t,D1)
plt.plot(-t,D2)

#%% Simulation of a model that responds in a nonlinear way to its inputs
# 1. Linear filtering stage
n_samples   = 500000
noise_stim  = np.random.normal(0.0,1.0,n_samples)
L1 = np.convolve(noise_stim,D1)[:n_samples]
L2 = np.convolve(noise_stim,D2)[:n_samples]

# Nonlinear stage
r_0 = 0    # Baseline firing rate [Hz]
r_max = 70 # maximam firing rate

F_L = np.square(L1) + np.square(L2)
F_L = r_max*(F_L/np.max(F_L))
r_est = r_0 + F_L

# Poisson spike generation
p = r_est*0.001
spikes = np.random.rand(n_samples)
spikes[spikes < p] = 1
spikes[spikes < 1] = 0

# Estimate the STA
spike_times = np.where(spikes == 1)[0]
sta = np.zeros(L)
for i in spike_times:
    if i < L:
        continue
    
    sta += noise_stim[(i-L):i]

sta = sta/np.sum(spike_times > L)
plt.plot(sta)