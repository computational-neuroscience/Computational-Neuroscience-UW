# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:53:46 2016

@author: Christophe
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the passive membrane model
R = 3
C = 0.01
tau = R*C
V_rest = 0.0

t = np.arange(0,0.5,0.001)

# The exact solution has to be calculated during  three different intervals
I_ext = 0.0
V_init = 0.0
V_1 = (V_rest + R*I_ext)*(1-np.exp(-t/tau)) + V_init*np.exp(-t/tau)

I_ext = 0.5
V_init = V_1[-1]
V_2 = (V_rest + R*I_ext)*(1-np.exp(-t/tau)) + V_init*np.exp(-t/tau)

I_ext = 0.0
V_init = V_2[-1]
V_3 = (V_rest + R*I_ext)*(1-np.exp(-t/tau)) + V_init*np.exp(-t/tau)

V_exact = np.concatenate((V_1,V_2,V_3))
I = np.concatenate((np.zeros(len(t)),0.5*np.ones(len(t)),np.zeros(len(t))))

# Euler solution
V_euler = np.zeros(len(I))
for i in range(1,len(I)):
    dV = (-(V_euler[i-1] - V_rest) + R*I[i])/tau
    V_euler[i] = V_euler[i-1] + dV*0.001

plt.subplot(1,2,1)
plt.plot(np.arange(0,1.5,0.001),V_exact,'k')
plt.plot(np.arange(0,1.5,0.001),I,'r')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.title('Exact solution')
plt.subplot(1,2,2)
plt.plot(np.arange(0,1.5,0.001),V_euler,'--k')
plt.plot(np.arange(0,1.5,0.001),I,'r')
plt.title('Euler approximation')
#%% Alpha and beta functions
# As given in http://icwww.epfl.ch/~gerstner/SPNM/node14.html
# These use a resting potential that corresponds to 0.0 mV
V = np.arange(-80.0,80,0.01)

alpha_n = (0.1 - 0.01*V)/(np.exp(1-0.1*V)-1)
alpha_m = (2.5 - 0.1*V)/(np.exp(2.5 - 0.1*V)-1)
alpha_h = 0.07*np.exp(-V/20.0)

beta_n  = 0.125*np.exp(-V/80.0)
beta_m  = 4.0*np.exp(-V/18.0)
beta_h   = 1.0/(np.exp(3 - 0.1*V) + 1)

tau_n = 1/(alpha_n + beta_n)
inf_n = alpha_n*tau_n

tau_m = 1/(alpha_m + beta_m)
inf_m = alpha_m*tau_m 

tau_h = 1/(alpha_h + beta_h)
inf_h = alpha_h*tau_h

plt.clf()
plt.subplot(1,2,1)
plt.plot(V,inf_n)
plt.plot(V,inf_m)
plt.plot(V,inf_h)
plt.title('Steady state values')
plt.xlabel('Voltage (mV)')
plt.subplot(1,2,2)
plt.plot(V,tau_n)
plt.plot(V,tau_m)
plt.plot(V,tau_h)
plt.title('Time constants')
plt.xlabel('Voltage (mV)')
#%% The full Hodgkin-Huxley model
import numpy as np
import matplotlib.pyplot as plt

# Equilibrium potentials
E_Na = 115.0
E_K  = -12.0
E_L  = 10.6

# Maximal conductances
g_Na = 120.0
g_K  = 36.0
g_L  = 0.3

dt = 0.01
T = 80
t = np.arange(0,T,dt)

V = np.zeros(len(t))
n = np.zeros(len(t))
m = np.zeros(len(t))
h = np.zeros(len(t))

I_E = 0.0
V[0] = 0.0
h[0] = 0.59
m[0] = 0.05
n[0] = 0.31

for i in range(1,len(t)):
    if i == 3000:
        I_E = 10.0
    if i == 3500:
        I_E = 0.0
        
    # Calculate the alpha and beta functions
    alpha_n = (0.1 - 0.01*V[i-1]) / ( np.exp(1   - 0.1*V[i-1]) - 1)
    alpha_m = (2.5 - 0.1 *V[i-1]) / ( np.exp(2.5 - 0.1*V[i-1]) - 1)
    alpha_h = 0.07*np.exp(-V[i-1]/20.0)
    
    beta_n = 0.125*np.exp(-V[i-1]/80.0)
    beta_m = 4.0 * np.exp(-V[i-1]/18.0)
    beta_h = 1 / ( np.exp(3 - 0.1*V[i-1]) + 1)
    
    # Calculate the time constants and steady state values
    tau_n = 1.0/(alpha_n + beta_n)
    inf_n = alpha_n*tau_n
    
    tau_m = 1.0/(alpha_m + beta_m)
    inf_m = alpha_m*tau_m
    
    tau_h = 1.0/(alpha_h + beta_h)
    inf_h = alpha_h*tau_h
    
    # Update the channel opening probabilities    
    n[i]  = (1-dt/tau_n)*n[i-1] + (dt/tau_n)*inf_n
    m[i]  = (1-dt/tau_m)*m[i-1] + (dt/tau_m)*inf_m
    h[i]  = (1-dt/tau_h)*h[i-1] + (dt/tau_h)*inf_h

    # Update the membrane potential equation
    I_Na = g_Na*(m[i]**3)*h[i]  * (V[i-1]-E_Na)
    I_K  = g_K *(n[i]**4)       * (V[i-1]-E_K)
    I_L  = g_L                  * (V[i-1]-E_L)
    
    #dv = -(I_Na + I_K + I_L - I_E)
    dv = I_E - (I_Na + I_K + I_L)
    V[i]  = V[i-1] + dv*dt
    
plt.clf()
plt.subplot(1,3,1)
plt.plot(t,V)
plt.subplot(1,3,2)
plt.plot(t,n)
plt.plot(t,m)
plt.plot(t,h)
plt.legend(['n','m','h'])
plt.subplot(1,3,3)
plt.plot(t,h * m**3)
plt.plot(t,n**4)
plt.legend(['Na','K'])