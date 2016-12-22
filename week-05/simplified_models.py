# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 23:11:39 2016

@author: Christophe
"""
import numpy as np
import matplotlib.pyplot as plt
#%% An integrate and fire model
V_thres =-0.01 # Threshold potential [V]
V_max   = 0.04 # Spike potential [V]
V_reset =-0.07 # Reset potential [V]
V_eq    =-0.06 # Resting potential [V]
a       = 25
dt = 0.001     # timestep [s]

# define time variable and voltage trace
t = np.arange(0,0.5,dt)
V_trace = np.zeros(len(t))
V_trace[0] = V_eq

# define a current trace 
I_max = 1.5
I_trace = np.zeros(len(t))
I_trace[200:400] = I_max

# Simulate the integrate and fire model
refrac = 0 # Keeps track of refractory period
for i in range(1,len(t)):
    V  = V_trace[i-1]
    dv = -a*(V - V_eq) + I_trace[i]
    
    if refrac > 0:
        V_trace[i] = V_reset
        refrac -= 1
        continue
        
    V_trace[i] = V + dv*dt
    
    if V_trace[i] > V_thres:
        V_trace[i] = V_max
        refrac = 3

# Calculate f(V)
V = np.arange(-0.08,0.06,0.01)
f_V = -a*(V-V_eq)

# Make a nice plot
plt.clf()
plt.subplot(1,2,1)
plt.plot(t,V_trace)
plt.axhline(V_thres,linestyle = '--',color='k')
plt.gca().set(xlabel='Time [s]',ylabel='Membrane potential [V]')
plt.title('Integrate and fire neuron')
plt.subplot(1,2,2)
plt.plot(V,f_V,'b')
plt.plot(V,f_V + I_max)
plt.axhline(0,color='k')
plt.axvline(0,color='k')
plt.plot(V_eq,0,'og')
plt.plot(V_eq + I_max/a,0,'og')
plt.plot(V_thres,0,'or')
plt.title('Phase plane')
plt.legend(['I = 0','I = ' + str(I_max)])
plt.gca().set(xlabel = 'V', ylabel ='f(V)')
#%% Exponential integrate and fire neuron
delta_exp = 0.037

# Prepare voltage trace
t = np.arange(0,0.5,dt)
V_trace = np.zeros(len(t))
V_trace[0] = V_eq

# define a current trace 
I_max = 1.9
I_trace = np.zeros(len(t))
I_trace[200:400] = I_max

# Simulate the exponential integrate and fire model
refrac = 0 # Keeps track of refractory period
for i in range(1,len(t)):
    V  = V_trace[i-1]
    dv = -a*(V - V_eq) + np.exp((V-V_thres)/delta_exp)*I_trace[i]
    
    if refrac > 0:
        V_trace[i] = V_reset
        refrac -= 1
        continue
        
    V_trace[i] = V + dv*dt
    
    if V_trace[i] > V_max:
        V_trace[i] = V_max
        refrac = 3
        
# Calculate f(V)
V = np.arange(-0.08,0.06,0.01)
f_V = -a*(V-V_eq) + np.exp((V - V_thres)/delta_exp)

plt.clf()
plt.subplot(1,2,1)
plt.plot(t,V_trace)
plt.subplot(1,2,2)
plt.plot(V,f_V)
plt.plot(V,f_V + I_max)
plt.axhline(0,color='k')
plt.axvline(0,color='k')

#%% Theta neuron

# Prepare trace variable
t = np.arange(0,50,dt)
theta_trace = np.zeros(len(t))
theta_trace[0] = 0.0

# define a current trace 
#I_max = -90.0
I_trace =-90.0*np.ones(len(t))
#I_trace = np.zeros(len(t))
#I_trace[200:400] = I_max

for i in range(1,len(t)):
    theta = theta_trace[i-1]

    dTheta = 1-np.cos(theta) + (1 + np.cos(theta))*I_trace[i]

    theta_trace[i] = (theta + dTheta*dt)%(2*np.pi)

plt.clf()
plt.plot(t,theta_trace)