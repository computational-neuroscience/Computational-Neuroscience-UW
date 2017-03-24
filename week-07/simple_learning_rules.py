# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:34:42 2017

@author: Christophe

Illustrates the effect of Hebbian learning,
Covariance learning and Oja's learning rule
"""

import numpy as np
import matplotlib.pyplot as plt


#%%  Hebbian learning
# Sample data parameters
u1_center = 0
u1_spread = 3.0

u2_center = 0
u2_spread = 10.0

n_dim     = 2
n_samples = 200


# Create samples from uniform distribution and rotating
theta= np.pi/4
rot_matrix = np.array([ [np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
U = np.array([np.random.normal(u1_center,u1_spread,n_samples),np.random.normal(u2_center,u2_spread,n_samples)])
U = rot_matrix.dot(U)

# Calculate the eigenvalues for the covariance matrix
E = np.cov(U)
[eigen_val,eigen_vec] = np.linalg.eigh(E)

#%% Illustration 1: weights grow without bounds

# Hebbian learning rule parameters
w_init = np.random.rand(2)-0.5
w = w_init.copy()
epsilon = 0.0005

# Run the update rule
weights   = np.zeros(n_samples)
alignment = np.zeros(n_samples)

for i in range(n_samples):
    u = U[:,i]
    v = w.dot(u)
    dw = epsilon*v*u
    w = w + dw
    
    weights[i] = np.linalg.norm(w)
    alignment[i]= np.dot(eigen_vec[:,1],w)/(np.linalg.norm(eigen_vec[:,1])*np.linalg.norm(w))

plt.subplot(1,2,1)
plt.plot(weights)
plt.xlabel('Sample presented')
plt.ylabel('|w|')
plt.title('Evolution of weight vector')

plt.subplot(1,2,2)
plt.plot(alignment)
plt.axhline(1,color='r')
plt.axhline(-1,color='r')
plt.ylim(-1.1,1.1)
plt.title('Alignment with covariane matrix eigenvector')
plt.xlabel('Sample presented')
plt.ylabel('cos(theta)')
#%% Illustration 2: weights align with the principal eigenvector of the input covariance matrix
w_normed = w/np.linalg.norm(w)
w_E = E.dot(w)/np.linalg.norm(E.dot(w))

r = 40
plt.plot(U[0,:],U[1,:],'x')
plt.xlim(-r,r)
plt.ylim(-r,r)
plt.axhline(0,color='k')
plt.axvline(0,color='k')

plt.arrow(0,0,r*w_normed[0],r*w_normed[1],fc='k',ec='k',head_width=2)
plt.arrow(0,0,r*w_E[0],r*w_E[1],fc='g',ec='g',head_width=2)
plt.xlabel(r'$u_1$')
plt.ylabel(r'$u_2$')
plt.title('Data and normed weight vector')