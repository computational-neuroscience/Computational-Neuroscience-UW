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


# Create samples from uniform distribution and rotates the data cloud
theta= np.pi/4
rot_matrix = np.array([ [np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
U = np.array([np.random.normal(u1_center,u1_spread,n_samples),np.random.normal(u2_center,u2_spread,n_samples)])
U = rot_matrix.dot(U)

# Calculate the eigenvalues for the covariance matrix
E = np.cov(U)
[eigen_val,eigen_vec] = np.linalg.eigh(E)

#%% Illustration 1: weights grow without bounds

# Hebbian learning rule parameters
w_init = np.random.rand(2)-0.5 # Initial parameters
w = w_init.copy()              # Copy of initial parameters
epsilon = 0.0005               # Learning rate

# Variables to store the weight vector norm and alignment with the first
# principal component of the input data
weights_norm   = np.zeros(n_samples)
alignment      = np.zeros(n_samples)

# Run the Hebbiean learning rule
for i in range(n_samples):
    # Apply
    u = U[:,i]
    v = w.dot(u)
    dw = epsilon*v*u
    w = w + dw
    
    weights_norm[i] = np.linalg.norm(w)
    alignment[i]= np.dot(eigen_vec[:,1],w)/(np.linalg.norm(eigen_vec[:,1])*np.linalg.norm(w))


# Plot weight vector norm and alignment with principal component
plt.subplot(1,2,1)
plt.plot(weights_norm)
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
w_normed = w/np.linalg.norm(w)          # Normed weight vector from Hebbian learning
w_E = E.dot(w)/np.linalg.norm(E.dot(w)) # Weight vector resulting from multiplying with covariance matrix
w_C = eigen_vec[:,-1]                   # Weight vector based on first eigenvector of covariance matrix

r = 40 # Constant for setting arrow lengths

arrow_1 = plt.arrow(0,0,r*w_normed[0],r*w_normed[1],fc='k',ec='k',head_width=2)
arrow_2 = plt.arrow(0,0,r*w_E[0],r*w_E[1],fc='g',ec='g',head_width=2)
arrow_3 = plt.arrow(0,0,r*w_C[0],r*w_C[1],fc='r',ec='r',head_width=2)

plt.plot(U[0,:],U[1,:],'x')
plt.xlim(-r,r)
plt.ylim(-r,r)
plt.axhline(0,color='k')
plt.axvline(0,color='k')

plt.xlabel(r'$u_1$')
plt.ylabel(r'$u_2$')
plt.title('Data and weight vectors')
plt.legend([arrow_1, arrow_2, arrow_3],['Hebbian weights','Covariance weights','Eigenvector'])