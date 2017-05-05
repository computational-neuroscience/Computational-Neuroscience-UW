# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:41:23 2017

@author: Christophe
In this example, a neural network is implemented with an input layer, a hidden
layer and an output layer. A classification problem is presented that solves
a nonlineary-separable category structure (i.e. they cannot be separated 
using a straight line). There are two categories, represented by a binary 
output node. In the plots, the category membership is color coded.
"""
import numpy as np
import matplotlib.pyplot as plt

# Define input data: 2 categories
n_clusters = 4
M = [[-1,-1],[-1,1],[1,-1],[1,1]]
sigma = 0.1
cluster_samples = [20, 20, 20, 20]
class_labels    = [0, 1, 1, 0]

cluster_data = np.ones((0,3))
for i in range(n_clusters):
    x = np.random.normal(M[i][0],sigma,cluster_samples[i])
    y = np.random.normal(M[i][1],sigma,cluster_samples[i])
    c = class_labels[i]*np.ones(cluster_samples[i])
    
    cluster_data = np.concatenate((cluster_data,np.array([x,y,c]).transpose()),axis=0)
    
#%% Inspect the input data
plt.scatter(cluster_data[:,0],cluster_data[:,1],c=cluster_data[:,2])
plt.clim(0,1)
plt.axhline(0,color='k', linewidth=0.5)
plt.axvline(0,color='k', linewidth=0.5)

#%% Implementation of the backpropagation algorithm
# Definition of the network topology. The number of input units equals
# 3 because an additional bias unit is included.
input_units  = 3
hidden_units = 10
output_units = 1

eps = 1.5
n_iterations= 20
beta = 1.5

w = np.random.normal(0.0,1.0,(hidden_units,input_units))
W = np.random.normal(0.0,1.0,(output_units,hidden_units))

def sigmoid(a, beta = 0.5):
    return 1/(1 + np.exp(-beta*a))

def dev_sigmoid(a, beta = 0.5):
    x = sigmoid(a,beta)
    return x*(1-x)

sse = np.zeros(n_iterations)
for it_index in range(n_iterations):
    for data_index in range(len(cluster_data)):
        x = np.concatenate(([1],cluster_data[data_index,0:2]))
        d = np.array([cluster_data[data_index,2]])
        
        # Forward pass 1: Calculate hidden layer activations
        u_sum = np.dot(w,x)
        u_act = sigmoid(u_sum,beta)
        
        # Forward pass 2: Calculate output layer activations
        v_sum = np.dot(W,u_act)
        v_act = sigmoid(v_sum,beta)
        
        # Backward pass 1: update output layer weights
        for i in range(output_units):
            dW = -(d[i]-v_act[i])*dev_sigmoid(v_sum[i],beta)*u_act
            W[i,:] -= eps*dW
        
        # Backward pass 2: update hidden layer weights
        for j in range(hidden_units):
            du_j = -np.sum( (d-v_act)*dev_sigmoid(v_sum)*W[:,j] )
            dw_k = dev_sigmoid(u_sum[j],beta)*x
            w[j,:] -= eps*du_j*dw_k
        
        sse[it_index] += np.sum(np.square(d-v_act))

sse /= len(cluster_data)
plt.plot(sse)

#%% Calculate the final predictions and plot the results
predictions = np.zeros(len(cluster_data))
for data_index in range(len(cluster_data)):
    x = np.concatenate(([1],cluster_data[data_index,0:2]))
    
    u = sigmoid(np.dot(w,x),beta)
    v = sigmoid(np.dot(W,u),beta)
    
    predictions[data_index] = v

plt.subplot(121)
plt.title('Actual cl)
plt.scatter(cluster_data[:,0],cluster_data[:,1],c=cluster_data[:,2])
plt.axhline(0,color='k', linewidth=0.5)
plt.axvline(0,color='k', linewidth=0.5)
plt.clim(0,1)
plt.subplot(122)
plt.scatter(cluster_data[:,0],cluster_data[:,1],c=predictions)
plt.axhline(0,color='k', linewidth=0.5)
plt.axvline(0,color='k', linewidth=0.5)
plt.clim(0,1)