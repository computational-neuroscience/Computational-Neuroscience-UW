# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:27:02 2017

@author: Christophe

In this script I apply the linear-nonlinear model of a neuron. The model contains
a linear filtering stage that is followed by a static nonlinearity that produces
a rate that can be plugged in a Poisson spike generator
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up some general parameters
dt = 0.0005  # Simulation step for Poisson spike generation
T  = 200      # Simulation duration in seconds
n_bins = int(T/dt) 

# Define a white noise stimulus
white_noise = np.random.normal(0,1,n_bins)

# Define the linear filter
filter_length = 0.1 # Filter length in seconds
filter_bins   = int(filter_length/dt)
filter_t      = np.arange(0,filter_length,dt)

linear_filter = np.sin(2*np.pi*10*filter_t)
filtered_noise = np.convolve(white_noise,linear_filter)

# Define the nonlinearity
def non_lin(x):
    if x < 2:
        return 0
    elif x > 10:
        return 10
    else:
        return x


         
# Feed into Poisson spike generator
spike_train = np.zeros(n_bins)
for i in range(n_bins):
    r = non_lin(filtered_noise[i])
    p = r*dt
    
    if np.random.rand() < p:
        spike_train[i] = 1


# Derive the linear filter using STA
idx = np.where(spike_train)[0]
sta_length = 200
sta = np.zeros(sta_length)

for st in idx:
    if st > sta_length:
        sta += white_noise[(st-sta_length):st]

sta /= len(idx)   

plt.plot(sta)                        
# Assess the type of the nonlinearity

