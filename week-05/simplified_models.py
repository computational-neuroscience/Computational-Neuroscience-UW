# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 23:11:39 2016

@author: Christophe
"""
import numpy as np
import matplotlib.pyplot as plt
#%% An integrate and fire model
R = 40.0  # Mohm
C = 1.0   # nF
tau = R*C # Membrane time constant

V_thres    = -40.0  # Threshold potential [mV]
V_max      =  40.0 # Spike potential     [mV]
V_reset    = -70.0 # Reset potential     [mV]
V_eq       = -60.0 # Resting potential   [mV]
max_refrac = 3     # time steps in refractory period

# define time variable and voltage trace
dt = 1     
t = np.arange(0,500,dt)
V_trace = np.zeros(len(t))
V_trace[0] = V_eq

# define a current trace 
I_max = 0.8
I_trace = np.zeros(len(t))
I_trace[200:400] = I_max

# Simulate the integrate and fire model
refrac = 0 # Keeps track of refractory period
for i in range(1,len(t)):
    V  = V_trace[i-1]
    dv = (-(V - V_eq) + R*I_trace[i])/tau
    
    if refrac > 0:
        V_trace[i] = V_reset
        refrac -= 1
        continue
        
    V_trace[i] = V + dv*dt
    
    if V_trace[i] > V_thres:
        V_trace[i] = V_max
        refrac = 3

# Plot the resulting voltage trace
plt.clf()
plt.subplot(1,2,1)
plt.plot(t,V_trace)
plt.axhline(V_thres,linestyle = '--',color='k')
plt.gca().set(xlabel='Time [ms]',ylabel='Membrane potential [V]')
plt.title('Integrate and fire neuron')


# Calculate the plot and the phase plane
V = np.arange(-80,60,1)
f_V = -(V-V_eq)/tau

plt.subplot(1,2,2)
plt.plot(V,f_V,'b')
plt.plot(V,f_V + R*I_max/tau,'g')
plt.axhline(0,color='k')
plt.axvline(0,color='k')
plt.plot(V_eq,0,'og')
plt.plot(V_eq + R*I_max,0,'og')
plt.plot(V_thres,0,'or')
plt.title('Phase plane')
plt.legend(['I = 0','I = ' + str(I_max)])
plt.gca().set(xlabel = 'Potential [mV]', ylabel ='f(V)')
#%% Exponential integrate and fire neuron
delta_exp = 7.0

# Prepare voltage trace
t = np.arange(0,200,dt)
V_trace = np.zeros((2,len(t)))
V_trace[:,0] = V_eq

# define a current trace 
I_max = [1.49,1.50]
I_trace = np.zeros(len(t))

for i in range(2):
    I_trace[50:95] = I_max[i]
    refrac = 0 
    for j in range(1,len(t)):
        if refrac > 0:
            V_trace[i,j] = V_reset
            refrac -= 1
            continue
    
        V  = V_trace[i,j-1]
        dv = (-(V - V_eq) + np.exp((V-V_thres)/delta_exp) + R*I_trace[j])/tau
        V_trace[i,j] = V + dv*dt
        
        if V_trace[i,j] > V_max:
            V_trace[i,j] = V_max
            refrac = max_refrac
        
# Plot voltage traces
plt.clf()
plt.subplot(1,2,1)
plt.plot(t,V_trace[0,:],'--k')
plt.plot(t,V_trace[1,:],'k')
plt.title('Voltage trace')
plt.axvline(95,color='r')
plt.gca().set(ylabel = 'Potential [mV]', xlabel ='Time [ms]')
# Calculate phase plane and set up phase plane plot
V = np.arange(-80.0,60.0,1.0)
f_V = (-(V-V_eq) + np.exp((V - V_thres)/delta_exp))/tau

fp_1 = -59.94
fp_2 = -13.05

plt.subplot(1,2,2)
plt.axhline(0,color='k')
plt.axvline(0,color='k')
plt.plot(V,f_V)
plt.plot(V,f_V + R*I_max[0]/tau)
plt.plot(fp_1,0,'og')
plt.plot(fp_2,0,'or')
plt.ylim(-5,10)
plt.title('Phase plane')
plt.gca().set(xlabel = 'Potential [mV]', ylabel ='dV/dt')
#%% Finding the roots of the exponential integrate and fire equation

x_0 = -13.0   # Starting value for the algorithm

from scipy.optimize import newton

def F(V):
    y = -( V - V_eq) + np.exp( (V-V_thres)/delta_exp)
    return y/tau
    
print (newton(F,x_0))
#%% THETA NEURON
#% Phase plane with external current
I      = -0.1
theta  = np.arange(0,2*np.pi,0.01)
dtheta = 1 - np.cos(theta) + (1 + np.cos(theta))*I

# Determine the fixed points of the system
fp_1 = np.arccos( -(1+I)/(-1+I))
fp_2 = 2*np.pi-fp_1

# Simulate the equations for two initial starting values
t = np.arange(0,30,dt)
theta_trace = np.zeros((2,len(t)))
theta_trace[:,0] = [fp_1-0.1,fp_1 + 0.1]

for i in range(2):
    for j in range(1,len(t)):
        current_theta = theta_trace[i,j-1]
    
        dTheta = 1-np.cos(current_theta) + (1 + np.cos(current_theta))*I
    
        theta_trace[i,j] = current_theta + dTheta*dt

plt.clf()
plt.subplot(1,3,1)
plt.title('Phase evolution')
plt.axhline(fp_1,color='r')
plt.axhline(fp_2,color='g')
plt.axhline(-fp_1,color='g')
plt.plot(t,theta_trace.transpose(),color='k')

plt.subplot(1,3,2)
plt.title('Unit circle representation')
circ=plt.Circle((0.0,0.0),radius=1,color='b', fill= False)
plt.gca().add_patch(circ)
plt.plot(np.cos(fp_1),np.sin(fp_1),'or')
plt.plot(np.cos(fp_2),np.sin(fp_2),'og')
plt.plot(-1,0,'ob')
plt.show()

plt.subplot(1,3,3)
plt.title('Phase plane')
plt.plot(theta,dtheta)
plt.axhline(0,color = 'k')
plt.plot(fp_1,0,'or')
plt.plot(fp_2,0,'og')