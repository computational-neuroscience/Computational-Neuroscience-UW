# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

'''
Returns cosine tuning curve for simulating M1 neuron responses
parameters: 
    s     = angle of the arm
    s_a   = prefered angle
    r_max = maximum firing rate
    r_0   = baseline firing rate
'''
def M1_response(s, s_a, r_max = 30.0, r_0 = 20.0):
    r = r_0 + (r_max - r_0)*np.cos(s - s_a)
    r[r<0] = 0
    return r

''' 
Returns rectified cosine tuning curve
following the description in Dayan and Abbott
'''
def cricket_response_1(s, s_a):
    r = np.cos(s-s_a)
    r[r<0] = 0
    return r
 
'''
Returns tuning curves for cricket cercal cells,
following the description in Salinas & Abbott (1994)
'''
def cricket_response_2(s, s_a, a = -0.14):
    r = (1/(1-a))*(np.cos(s-s_a)-a)
    r[r<0] = 0
    return r
    
#%% Plot the tuning curves of cercal interneurons
theta = np.deg2rad(np.array([45, 135, -135, -45]))
c_a = np.array([np.cos(theta),np.sin(theta)])

wind_direction = np.arange(-180,180,5)
tuning_curves = np.zeros((len(theta),len(wind_direction)))
ls = ['--','-']
for i in range(len(theta)):
    tuning_curves[i] = 25*cricket_response_1(np.deg2rad(wind_direction),theta[i])
    plt.plot(wind_direction,tuning_curves[i], 'k',linestyle = ls[i%2])

plt.title('Cricket cercal system tuning curves')
plt.xlabel('Stimulus (degrees)')
plt.ylabel('Firing rate')    
#%% For different wind directions simulate several trials and calculate the 
# squared difference with the true wind direction
n_trials = 100
error = np.zeros((n_trials,len(wind_direction)))

for i,s in enumerate(np.deg2rad(wind_direction)):    
    for trial_id in range(n_trials):
        r = 50*cricket_response_1(s,theta)
        r = r + np.random.normal(0,5,len(theta))
        r[r<0] = 0
        
        v_pop = np.zeros(2)
        for j in range(len(theta)):
            v_pop += r[j]*c_a[:,j]
        
        v_true = np.array([np.cos(s),np.sin(s)])
        v_pop = v_pop/np.linalg.norm(v_pop)
        theta_diff = np.arccos(np.dot(v_true,v_pop))
        error[trial_id,i] = np.square(np.rad2deg(theta_diff))

mean_error = np.sqrt(np.nanmean(error,0))
plt.plot(wind_direction,mean_error)
plt.xlim(-180,180)
plt.plot(np.rad2deg(theta),np.min(mean_error)*np.ones(len(theta)),'rv')
plt.xlabel('Stimulus (degrees)')
plt.ylabel('Average error')
plt.title('Population decoding of cercal interneurons')
#%% Plot the tuning curves of different motor neurons
separation = 15
prefered_directions = np.deg2rad(np.arange(0, 360, separation))
arm_direction = np.arange(0,360,5)
for i in range(len(prefered_directions)):
    plt.plot(arm_direction, M1_response(np.deg2rad(arm_direction), prefered_directions[i]),'-k')

plt.title('M1 tuning curves')
plt.xlabel('Direction (degrees)')
plt.ylabel('Firing rate')
plt.xlim(0, 360)

#%%
r_max = 50.0
r_0 = 20.0
n_trials = 100
c_a = np.array([np.cos(prefered_directions),np.sin(prefered_directions)])

n_repetitions = 100
error = np.zeros((n_repetitions,len(arm_direction)))
for i, s in enumerate(np.deg2rad(arm_direction)):
    v_pop = np.zeros(2)
    for rep in range(n_repetitions):
        rate = M1_response(s, prefered_directions, r_max,r_0)
        spike_count = np.random.poisson(rate)
        
        v_pop += np.sum(((spike_count-r_0)/r_max)*c_a,1)
        
        v_true = np.array([np.cos(s),np.sin(s)])
        v_pop = v_pop/np.linalg.norm(v_pop)
        theta_diff = np.arccos(np.dot(v_true,v_pop))
        error[rep,i] = np.square(np.rad2deg(theta_diff))

mean_error = np.sqrt(np.nanmean(error,0))
plt.plot(arm_direction,mean_error)
plt.xlim(-0,360)
plt.xlabel('Stimulus (degrees)')
plt.ylabel('Average error')
plt.title('Population decoding of M1 motor neurons')