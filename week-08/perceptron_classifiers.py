# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 19:37:16 2017

@author: Christophe
"""

import matplotlib.pyplot as plt
import numpy as np

#%% Perceptron model
# Simulates two data clouds samples from two different uniform distributions
# Then runs the perceptron learning algorithm to obtain a line that separates the
# Generate some random data
n_samples= 30
cluster_A = np.random.uniform(-2.0,-1.0,(n_samples,2))
cluster_B = np.random.uniform(-0.5, 1.5,(n_samples,2))
target_values = np.concatenate((np.ones((n_samples,1)),-1*np.ones((n_samples,1))))

data_points = np.concatenate((cluster_A,cluster_B),0)
data_points = np.concatenate((data_points,target_values),1)

# Run the perceptron learning algorithm
w_1 = np.random.randn(1,2)
u_1 = np.random.randn(1,1)
eps = 0.012
n_steps = 10
training_error = np.zeros(n_steps)

for training_step in range(n_steps):
    for p in data_points:
        v = np.sign(w_1.dot(p[:2]) - u_1)
        v_d = p[2]
        
        delta_w = eps*(v_d-v)*p[:2]
        delta_u = -eps*(v_d-v)
        
        w_1 += delta_w
        u_1 += delta_u
        
        training_error[training_step] += (v_d-v)**2/(n_samples*2)

# Compute line for plotting the separating hyperplane
x_0 = -2.5; x_1 = 2.0
y_0 = np.squeeze(u_1/w_1[0,1] - w_1[0,0]*x_0/w_1[0,1])
y_1 = np.squeeze(u_1/w_1[0,1] - w_1[0,0]*x_1/w_1[0,1])

# Produce some plots
plt.clf()
plt.subplot(1,2,1)
plt.scatter(cluster_A[:,0],cluster_A[:,1],color = 'r')
plt.scatter(cluster_B[:,0],cluster_B[:,1],color = 'g')
plt.axhline(0,color='k')
plt.axvline(0,color='k')
plt.plot([x_0,x_1],[y_0,y_1],color = 'k')
plt.title('Input space')
plt.xlim((-2.0,1.5))

plt.subplot(1,2,2)
plt.plot(training_error)
plt.title('Training error')
plt.xlabel('Training step')
plt.ylabel('SSE')

#%% Multilayer perceptron model for the XOR function
input_array = np.array([ [ 1.0, 1.0],
                         [ 1.0,-1.0],
                         [-1.0, 1.0],
                         [-1.0,-1.0]])
T = np.array([1.0, -1.0, -1.0, 1.0])

u_h = -1
u_o = 1
w_ih = np.array([ 1.0, 1.0])
w_io = np.array([-1.0,-1.0])
w_ho = np.array([ 2.0])

for i in range(len(T)):
    h = np.sign(w_ih.dot(input_array[i])-u_h)
    o = np.sign(w_io.dot(input_array[i]) + w_ho*h - u_o)
    print("Desired output: " + str(T[i]) + ", computed output: " + str(o[0]))