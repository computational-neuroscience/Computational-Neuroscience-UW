# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:34:42 2017

@author: Christophe

Simulation of expectation maximization algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#%%  EM algorithm
# Define cluster parameters (we sample data from 2D Gaussian clusters)
c1_center = np.array([5,7])      # Cluster 1 mean
c1_spread = np.array([ 4, 4])    # Cluster 1 standard deviation
 
c2_center = np.array([-1, -3])   # Cluster 2 mean
c2_spread = np.array([ 2, 2])    # Cluster 2 standard deviation
            
n_samples = 100        # Samples in each cluster

# Generate data for each cluster
c1_data = np.array([np.random.normal(c1_center[0],c1_spread[0],n_samples),np.random.normal(c1_center[0],c1_spread[1],n_samples)])
c2_data = np.array([np.random.normal(c2_center[0],c2_spread[1],n_samples),np.random.normal(c2_center[1],c2_spread[1],n_samples)])
data    = np.concatenate((c1_data,c2_data),axis=1)
plt.plot(c1_data[0,:],c1_data[1,:],'.k')
plt.plot(c2_data[0,:],c2_data[1,:],'.b')

# EM algorithm parameters
n_iterations = 10

# Initial cluster parameter estimates
c1_center_est = [ -5, 0]
c1_spread_est = [ 1, 1]

c2_center_est = [ 5, 0]
c2_spread_est = [ 1, 1]

# Run the algorithm
center_data = np.zeros((n_iterations,2,2))
spread_data = np.zeros((n_iterations,2,2))

for i in range(n_iterations):
    # Store current estimates
    center_data[i,:,0] = c1_center_est
    center_data[i,:,1] = c2_center_est
               
    spread_data[i,:,0] = c1_spread_est
    spread_data[i,:,1] = c2_spread_est
    
    # Expectation of the data
    p_1 = multivariate_normal.pdf(data.transpose(), mean=c1_center_est,cov = c1_spread_est)
    p_2 = multivariate_normal.pdf(data.transpose(), mean=c2_center_est,cov = c2_spread_est)
    
    # Use Bayes theorem to assign data to clusters
    cluster = p_1 > p_2
    
    # Update the parameters based on cluster assignment
    c1_center_est = np.mean(data[:,cluster==True],axis=1)
    c1_spread_est = np.var(data[:,cluster==True],axis=1)
    c2_center_est = np.mean(data[:,cluster==False],axis=1)
    c2_spread_est = np.var(data[:,cluster==False],axis=1)


# Grid for evaluating the probability distributions
[X,Y] = np.meshgrid(np.linspace(-10,20,50),np.linspace(-10,20,50))
a = np.array([X.flatten(),Y.flatten()]).transpose()

# Get the initial distribution
z_1    = np.reshape(multivariate_normal.pdf(a, mean = center_data[0,:,0], cov = spread_data[0,:,0]), (50,50))
z_2    = np.reshape(multivariate_normal.pdf(a, mean = center_data[0,:,1], cov = spread_data[0,:,1]), (50,50))
z_init = 0.5*z_1 + 0.5*z_2

# Get the final distribution
z_1     = np.reshape(multivariate_normal.pdf(a, mean = center_data[-1,:,0], cov = spread_data[-1,:,0]), (50,50))
z_2     = np.reshape(multivariate_normal.pdf(a, mean = center_data[-1,:,1], cov = spread_data[-1,:,1]), (50,50))
z_final = 0.5*z_1 + 0.5*z_2

# Get the actual distribution
z_1     = np.reshape(multivariate_normal.pdf(a, mean = c1_center, cov = c1_spread), (50,50))
z_2     = np.reshape(multivariate_normal.pdf(a, mean = c2_center, cov = c2_spread), (50,50))
z_final = 0.5*z_1 + 0.5*z_2

# Plot the different distributions
plt.subplot(1,3,1)
plt.imshow(z_init,extent=[-10,20,20,-10],alpha=0.9,cmap='Reds')
plt.title('Initial estimate')
plt.subplot(1,3,2)
plt.imshow(z_final,extent=[-10,20,20,-10],alpha=0.9,cmap='Reds')
plt.title('Final estimate')
plt.subplot(1,3,3)
plt.imshow(z_final,extent=[-10,20,20,-10],alpha=0.9,cmap='Reds')
plt.title('Actual distribution')
