# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 21:48:25 2017

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

# Temporal difference learning rule

T        = 200   # Total number of time steps
t_stim   = 50   # stimulus delivery time
t_reward = 150   # reward delivery time

u = np.zeros(T) # State of the environment
r = np.zeros(T) # Total future reward
w = np.zeros(T) # weight function for past environment states
v = np.zeros(T) # Expected future reward

u[t_stim]   = 1
r[t_reward-2:t_reward+2] = 1


plt.clf()
plt.subplot(4,2,1)
plt.plot(u)
plt.subplot(4,2,3)
plt.plot(r)
#%% single trial: single-step learning
eps = 0.9
w = np.zeros(T) # weight function for past environment states
n_trials = 100
err_evolution = np.zeros((n_trials,T-1))
for trial_id in range(n_trials):
    v = np.zeros(T)
    for t in range(0,T-1):
        v[t]      = w[0:(t+1)].dot(u[t::-1])
        v[t+1]    = w[0:(t+2)].dot(u[t+1::-1])
        delta_err = r[t] + v[t+1] - v[t]
        err_evolution[trial_id,t] = delta_err

        w[0:(t+1)] += eps*delta_err*u[t::-1]
        
    
plt.clf()
plt.subplot(1,2,1)
plt.imshow(err_evolution, origin='lower')
plt.xlabel('Time')
plt.ylabel('Trial')
plt.title('Prediction error evolution')
plt.subplot(4,2,2),
plt.plot(u,'g')
plt.plot(r,'r')
plt.yticks([],[])
plt.title('Stimulus (green) and reward (red)')

plt.subplot(4,2,4),
plt.plot(v)
plt.yticks([],[])
plt.ylabel('v[t]')

plt.subplot(4,2,6),
plt.plot(np.diff(v))
plt.yticks([],[])
plt.ylabel(r'$\Delta v[t]$')

plt.subplot(4,2,8),
plt.plot(err_evolution[-1,:])
plt.yticks([],[])
plt.ylabel(r'$\delta$')
plt.xlabel('Time')