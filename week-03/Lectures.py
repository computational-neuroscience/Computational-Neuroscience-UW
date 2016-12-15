# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 18:17:53 2016

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Likelihood ratio test
# Function for generating values from a gaussian probability density function
def gauss_pdf(x,mu,sigma):
    return  np.exp( -((x-mu)**2)/(2*sigma**2))/(np.sqrt(2*np.pi*sigma**2))

# Distribution parameters
mu_1 = 5
mu_2 = 7

sigma_1 = 0.5
sigma_2 = 1

# Generate the distributions
x = np.arange(3,14,0.01)
d1 = gauss_pdf(x,mu_1,sigma_1)
d2 = gauss_pdf(x,mu_2,sigma_2)

# Calculate the likelihood ratio
lr = d2/d1
lr_1   = np.argmax(lr>=1)

# Plot the results
plt.subplot(121)
plt.plot(x,d1)
plt.plot(x,d2)
plt.axvline(x[lr_1],linewidth=2,color='r')
plt.legend(['S+','S-'])
plt.title('Response distribution')
plt.subplot(122)
plt.plot(x[:500],lr[:500])
plt.axhline(y=1,linewidth = 2, color='r')
plt.ylim(0,2)
plt.title('Likelihood ratio')

#%% Illustrates how the likelihood function can be sampled repeatedly
# This amounts to collecting evidence. A decision about th stimulus can be made
# after the accumulated evidence reaches some threshold

n_samples = 100
evidence = np.zeros((3,n_samples))
for i in range(1,n_samples):
    sample = np.random.randint(200,400)
    evidence[0,i] = evidence[0,i-1] + np.log(lr[sample])
    evidence[1,i] = evidence[1,i-1] + np.log(lr[sample+100])
    evidence[2,i] = evidence[2,i-1] + np.log(lr[sample+200])
    
plt.plot(evidence.transpose())
plt.ylim(-50,50)
plt.title('Accumlation of evidence')
plt.xlabel('Time step')
plt.ylabel('Accumulated evidence')

#%% Illustrates that scaling that taking into account prior information about the
# stimulus leads to a rescaling of the response distributions and subsequently
# a different respond threshold value
p_1 = 1
p_2 = 0.1

pd1 = p_1*d1
pd2 = p_2*d2
lr_1 = np.argmax((d2/d1) >= 1)
lr_2 = np.argmax((pd2/pd1) >= 1)

plt.subplot(121)
plt.plot(x,d1)
plt.plot(x,d2)
plt.axvline(x[lr_1],linewidth=2,color='r')
plt.title('Original distribution')
plt.subplot(122)
plt.plot(x,pd1)
plt.plot(x,pd2)
plt.axvline(x[lr_2],linewidth=2,color='r')
plt.title('Scaled distribution')

#%% Finally, we can take into account penalties for making mistakes incorrect decisions
# This leads to a shift in the response threshold. Taking into account penalties again
# leads to a rescaling of the response distributions
L_2 = 2.0
L_1 = 1.0

lr_1   = np.argmax(lr>=1)
lr_2   = np.argmax(lr>=L_2/L_1)

plt.subplot(121)
plt.plot(x,d1)
plt.plot(x,d2)
plt.axvline(x[lr_1],linewidth=2,color='r')
plt.legend(['S-','S+'])
plt.title('Original distribution')
plt.subplot(122)
plt.plot(x,d1/L_1)
plt.plot(x,d2/L_2)
plt.axvline(x[lr_2],linewidth=2,color='r')
plt.legend(['S-','S+'])
plt.title('Response penalties')

#%% Illustration of Bayesian inference
# Function for producing values from gaussian curves at location x with mean
# equal to x_0, standard deviation s and scaling factor r_max
def gauss_tuning(x, x_0,s,r_max):
    return r_max*np.exp(-0.5*((x-x_0)/s)**2)

s = np.arange(0,2*np.pi,0.01)         # Continuous dimension to which neurons are tuned
sigma = 1                             # Standard deviation of the tuning curves
r_max   = 20                          # Firing rate for each neuron
n_neurons = 10                        # Number of simulated neurons
r_0 = np.linspace(0,2*np.pi,n_neurons)# Prefered orientation for each neuron 

# Produces a plot of the orientation tuning curves for each of the neurons in
# our population
cell_tuning = np.zeros( (n_neurons, len(s)))
for i in range(n_neurons):
    cell_tuning[i,:] = gauss_tuning(s,r_0[i],sigma,r_max) 
plt.plot(s, cell_tuning.transpose(),color='k')
plt.xlabel('Orientation (rad)')
plt.ylabel('Firing rate')
plt.title('Tuning curves')

#%% We can use the firing rates predicted from the tuning curves to simulate spike trains
# for each neuron in response to a particular orientation
T = 0.5                  # Interval during which stimulus is presented
s_index = 300            # Index of the stimulus that we show
n_repetitions = 10       # How many times we present the stimulus
t = np.arange(0,T,0.001) # Time points for simulation
neuron_response = np.zeros((n_repetitions*n_neurons,len(t)))
neuron_spike_count = np.zeros((n_repetitions,n_neurons))

for neuron_index in range(n_neurons):
    r_a = gauss_tuning(s[s_index],r_0[neuron_index],sigma,r_max)
    P = r_a*0.001
    for repetition_index in range(n_repetitions):
        time_bins = np.random.rand(len(t))
        time_bins[time_bins < P] = 1
        time_bins[time_bins < 1] = 0
        neuron_response[neuron_index*n_repetitions + repetition_index,:] = time_bins
        neuron_spike_count[repetition_index,neuron_index] = sum(time_bins)

# Predict response based on mean spike count
mean_response = np.mean(neuron_spike_count,0)
num = 0.0
denom = 0.0
for i in range(n_neurons):
    num += mean_response[i]*r_0[i]/(sigma**2)
    denom += mean_response[i]/(sigma**2)
s_pred = num/denom
print "Presented stimulus: " + str(s[s_index])
print "Predicted stimulus: " + str(s_pred)
plt.subplot(121)            
plt.spy(neuron_response,aspect='auto')
for i in np.arange(9.5,n_repetitions*n_neurons,n_repetitions):
    plt.axhline(i,color='k')
plt.xticks(range(0,len(t),100),t[range(0,len(t),100)])
plt.yticks(range(5,n_repetitions*n_neurons,n_repetitions),range(1,11))
plt.title('Raster plot')
plt.xlabel('Time (s)')
plt.ylabel('Cell number')
plt.subplot(122)
plt.plot(mean_response)
plt.xlabel('Cell number')
plt.ylabel('Mean spike count')
