# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:26:24 2017

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Preliminary code
dt = 0.01
t = np.arange(0, 2, dt)

f = 0.5
A = 0
stim = A + np.sin(2*np.pi*f*t) + np.random.normal(0.0, 0.05, len(t))
#%% Case 1a: Linear amplification of past input using a for loop
response = np.zeros(len(stim))
delay = 50
amplification = 1.5

for i in range(delay,len(stim)):
    response[i] = amplification*stim[i-delay]

plt.clf()
plt.plot(t,stim)
plt.plot(t,response)
plt.xlabel('Time (s)')
plt.legend(['Stimulus','response'])

#%% Case 1b: Linear amplification using convolution
lf = np.zeros(50)
lf[-1] = amplification
response = np.convolve(stim,lf)

plt.clf()
plt.plot(t,stim)
plt.plot(t,response[:len(t)])
plt.xlabel('Time (s)')
plt.legend(['Stimulus','response'])

#%% Case 2a: Running average 
filter_range = 0.4
filter_points= int(filter_range/dt)
filter_decay = 20.0

ra_filter = np.ones(filter_points)/filter_points
la_filter = np.exp(-filter_decay * np.arange(0,filter_range,dt))
la_filter = la_filter/np.sum(la_filter)

response = np.zeros((2,len(stim)))

for i in range(filter_points, len(stim)):
    for j in range(filter_points):
        response[0,i] += stim[i-j]*ra_filter[j]
        response[1,i] += stim[i-j]*la_filter[j]
            

plt.plot(ra_filter)
plt.axhline(0,color ='k')
plt.gca().invert_xaxis()
plt.subplot(1,2,2)
plt.plot(t, stim)
plt.plot(t, response[0,:])
plt.axis('off')
#%% Case 3a: Leaky average filter
la_filter = np.exp(-filter_decay * np.arange(0,filter_range,dt))
la_filter = la_filter/np.sum(la_filter)

response = np.zeros(len(stim))

for i in range(filter_points, len(stim)):
    for j in range(filter_points):
        response[i] += stim[i-j]*la_filter[j]

plt.subplot(1,2,1)
plt.plot(la_filter)
plt.axhline(0,color ='k')
plt.gca().invert_xaxis()
plt.subplot(1,2,2)
plt.plot(t, stim)
plt.plot(t, response)
plt.axis('off')
#%% Case 3: spatial filtering on-off ganglion cells
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

import scipy.misc
from scipy import signal
ascent = scipy.misc.ascent()
plt.gray()
plt.imshow(ascent)

a = signal.convolve2d(ascent, rgc)
plt.imshow(a)
#%% Case 4: Linear filter for V1 cells
[X,Y] = np.meshgrid(np.arange(-3,3,0.1),np.arange(-3,3,0.1))

f     = 0.4     # Spatial frequency
theta = 0.5     # Orientation
phi   = np.pi/2 # Phase offset
sigma = 0.7     # Standard deviation of the gaussian kernel
gamma = 0.9     # Aspect ratio

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


#%% Time reversal filter described by Adelson and Bergen (1985)
alpha = 1/15.0
t = np.arange(0,300,1)

D = alpha*np.exp(-alpha*t)*( ((alpha*t)**5)/(np.math.factorial(5)) - ((alpha*t)**7)/np.math.factorial(7))

plt.plot(-t,D)

