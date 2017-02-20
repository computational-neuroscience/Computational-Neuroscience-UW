# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:26:24 2017

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Preliminary code
dt = 0.01
time_points = np.arange(0, 2*np.pi, dt)

f = 0.2
A = 1
stim = A + np.sin(2*np.pi*f*time_points)
stim = stim + np.random.normal(0.0, 0.05, len(stim))

#%% Case 1: Linear amplification of past input
response = np.zeros(len(stim))
delay = 50
amplification = 1.2

for i in range(len(stim)):
    if i >= delay:
        response[i] = amplification*stim[i-delay]

#sv = np.convolve(s,lr,mode='same')

plt.clf()
plt.plot(time_points,stim)
plt.plot(time_points,response)
plt.xlabel('Time')
plt.legend(['Stimulus','response'])
#%% Case 2: Running average and leaky average filter
filter_range = 0.4
filter_points= int(filter_range/dt)
filter_decay = 20.0

ra_filter = np.ones(filter_points)/filter_points
la_filter = np.exp(-filter_decay * np.arange(0,filter_range,dt))
la_filter = la_filter/np.sum(la_filter)

response = np.zeros((2,len(stim)))

for i in range(len(stim)):
    if i >= filter_points:
        for j in range(filter_points):
            response[0,i] += stim[i-j]*ra_filter[j]
            response[1,i] += stim[i-j]*la_filter[j]
            
plt.subplot(1,2,1)
plt.plot(ra_filter)
plt.axhline(0,color ='k')
plt.gca().invert_xaxis()
plt.subplot(1,2,2)
plt.plot(time_points, stim)
plt.plot(time_points, response[0,:])
plt.axis('off')

plt.subplot(1,2,1)
plt.plot(la_filter)
plt.axhline(0,color ='k')
plt.gca().invert_xaxis()
plt.subplot(1,2,2)
plt.plot(time_points, stim)
plt.plot(time_points, response[1,:])
plt.axis('off')

#%% Case 3: spatial filtering on-off cells and V1 cells
[X,Y] = np.meshgrid(np.arange(-3,3,0.1),np.arange(-3,3,0.1))
B = 0.9
sigma_c = 0.5
sigma_s = 0.7

rgc = (1/(2*np.pi*sigma_c**2))*np.exp(- (X**2 + Y**2)/(2*sigma_c**2)) - (B/(2*np.pi*sigma_s**2))*np.exp(- (X**2 + Y**2)/(2*sigma_s**2))

plt.subplot(1,2,1)
plt.imshow(rgc)
plt.title('On center ganglion cell')
plt.axis('off')
plt.subplot(1,2,2)
plt.plot(rgc[30,:])
plt.axhline(0,color='k')
plt.title('Receptive field cross section')
plt.axis('off')
#%% Case 4: Linear filter with static nonlinearity
[X,Y] = np.meshgrid(np.arange(-3,3,0.1),np.arange(-3,3,0.1))

f     = 0.4
theta = 0.0
phi   = np.pi/2
sigma = 0.6
gamma = 0.8

X_rot = X*np.cos(theta) + Y*np.sin(theta)
Y_rot = -X*np.sin(theta) + Y*np.cos(theta)

g = np.exp( - (X_rot**2 + (gamma**2)*(Y_rot**2))/(2*sigma**2))*np.cos(2*np.pi*f*X_rot + phi)
plt.subplot(1,2,1)
plt.title('Simple cell RF')
plt.axis('off')
plt.imshow(g)
plt.subplot(1,2,2)
plt.plot(g[30,:])
plt.title('RF cross-section')
plt.axis('off')
plt.axhline(0,color='k')


#%% Generate temporal difference filter
#
sigma = 2
f = 0.3
x = np.arange(0, 5, 0.1) # Time is in ms, so generates a 5 ms wide filter
td_filter = np.exp(-(x**2)/sigma**2)*np.sin(2*np.pi*f*x)
td_filter = td_filter/np.sum(td_filter)
plt.plot(td_filter)

white_noise    = np.random.normal(0.0,1.0,200000) # Generates 10 seconds of data
filtered_noise = np.convolve(white_noise, td_filter)

plt.subplot(2,1,1)
plt.plot(white_noise)
plt.subplot(2,1,2)
plt.plot(filtered_noise)
#%% Case study
# Implements an Izhikevich spiking neuron that receives input proportional to receptive 
T = len(white_noise)
dt = 0.1

a = 0.02
b= 0.2
c = -55
d = 8

v = -70
u = -13

I = 10

result = np.zeros(T)
spikes = np.zeros(T)
for i in range(T):
    dv = 0.04*v**2 + 5*v + 140 - u + 25*filtered_noise[i]
    du = a*(b*v - u)
    
    v += dv*dt
    u += du*dt
    
    if v > 30:
        v = c
        u += d
        spikes[i] = 1
    
    result[i] = v
          
plt.plot(np.arange(0,T)*dt,result)
plt.xlabel('Time (ms)')

#%% Calculate STA
# Find indices where a spike has been triggered
spike_indices = np.where(spikes == 1)[0]

sta_win_length = 200
sta = np.zeros(sta_win_length)
for spike_index in spike_indices:
    if spike_index < sta_win_length:
        continue
    sta += filtered_noise[spike_index-sta_win_length:spike_index]
    
plt.plot(sta)