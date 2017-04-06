# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:34:42 2017

@author: Christophe

Simulation of competitive learning
"""

import numpy as np
import matplotlib.pyplot as plt


#%%  Competitive learning
# Define cluster parameters (we sample data from 2D Gaussian clusters)
c1_center = [10,10]    # Cluster 1 mean
c1_spread = [ 4, 4]    # Cluster 1 spread
 
c2_center = [-5, -15]  # Cluster 2 mean
c2_spread = [ 4, 4]    # Cluster 2 spread
            
n_samples = 50         # Samples in each cluster

# Generate data for each cluster
c1_data = np.array([np.random.normal(c1_center[0],c1_spread[1],n_samples),np.random.normal(c1_center[0],c1_spread[1],n_samples)])
c2_data = np.array([np.random.normal(c2_center[0],c2_spread[1],n_samples),np.random.normal(c2_center[1],c2_spread[1],n_samples)])
data    = np.concatenate((c1_data,c2_data),axis=1)


# Initialize random weights for two clusters
n_clusters = 2
weights    = np.random.normal(size=(n_clusters,n_clusters))

# Parameters for the competitive learning algorithm
epsilon = 0.05
n_trials= 200

# Competitive learning algorithm
weight_mov = np.zeros((n_trials,2,2)) # Keeps track of the weights on each trial
for i in range(n_trials):
    weight_mov[i,:,:] = weights
    
    # Determine the output activation of each neuron              
    sample_index = np.random.randint(0,2*n_samples)
    u = data[:,sample_index]
    v = weights.dot(u)
    
    # Determine the winning neuron
    winner  = np.argmax(v)
    delta_w = epsilon*(u-weights[winner,:])
    
    # Update the weights
    weights[winner,:] += delta_w

# Plot the data cloud
plt.plot(c1_data[0,:],c1_data[1,:],'xk',alpha = 0.5)
plt.plot(c2_data[0,:],c2_data[1,:],'xb',alpha = 0.5)
plt.xlim(-30,30)
plt.ylim(-30,30)

# Plot initial and final weights
plt.plot(weights[0,0],weights[0,1],'or')
plt.plot(weights[1,0],weights[1,1],'og')
plt.plot(weight_mov[:,0,0],weight_mov[:,0,1],'-.r')
plt.plot(weight_mov[:,1,0],weight_mov[:,1,1],'-.g')