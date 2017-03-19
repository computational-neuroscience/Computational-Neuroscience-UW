# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:08:58 2017

@author: Christophe

Simulation of Britten et al '92 data

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

def r(theta, theta_p = 0, c = 1.0, r_max = 30, K = 20, sigma = 10):
    A = c*r_max
    return A*np.exp(-0.5*(theta-theta_p)**2/(2*sigma**2)) + K
    
# Simulate n_trials with the preferred and the orthogonal stimulus
n_trials = 500
r_plus = np.random.poisson(r(0, c=1), n_trials)
r_min  = np.random.poisson(r(90,c=1), n_trials)  

bins = np.arange(np.min(r_min),np.max(r_plus))
plt.hist(r_plus,bins, alpha = 0.7)
plt.hist(r_min,bins, alpha = 0.7)

# Get mean response for s+ and s- stimulus
m_plus = r(0)
m_min  = r(90)
m_det  = (m_plus+m_min)/2.0
#%% Construct the ROC curve
z_values = np.arange(0,100,5)
alpha_z  = np.zeros(len(z_values))
beta_z   = np.zeros(len(z_values))

for i in range(len(z_values)):
    alpha_z[i] = float(np.sum(r_min>z_values[i]))/n_trials
    beta_z[i]  = float(np.sum(r_plus>z_values[i]))/n_trials

plt.plot(alpha_z, beta_z,'o-')
plt.plot([0,1],[0,1],'--k')
plt.axis('scaled')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
np.trapz(beta_z[::-1], x = alpha_z[::-1])

#%% Construct neurometric curve by simulating with different coherence values
c_values = np.arange(0,1.1,0.1)
auc_performance =np.zeros(len(c_values))

for i in range(len(c_values)):
    r_plus = np.random.poisson(r(0, c=c_values[i]), n_trials)
    r_min  = np.random.poisson(r(90,c=c_values[i]), n_trials)  
    
    alpha_z  = np.zeros(len(z_values))
    beta_z   = np.zeros(len(z_values))
    for j in range(len(z_values)):
        alpha_z[j] = float(np.sum(r_min>z_values[j]))/n_trials
        beta_z[j]  = float(np.sum(r_plus>z_values[j]))/n_trials

    auc_performance[i] = np.trapz(beta_z[::-1], x = alpha_z[::-1])
    
plt.plot(c_values, auc_performance)
plt.xlabel('Coherence level')
plt.ylabel('Performance')

#%% Construct neurometric curve by simulating a 2AFC task and a threshold 
# detection task
c_values = np.arange(0,1.1,0.1)
fc_performance =np.zeros(len(c_values))
det_performance = np.zeros(len(c_values))
for i in range(len(c_values)):
    r_plus = np.random.poisson(r(0, c=c_values[i]), n_trials)
    r_min  = np.random.poisson(r(90,c=c_values[i]), n_trials)  
    
    #m_det  = (r(0, c=c_values[i])+r(90,c=c_values[i]))/2.0

    fc_performance[i] = np.sum(r_plus>r_min)/float(n_trials)
    det_performance[i] = (np.sum(r_plus>m_det) + np.sum(r_min<m_det))/(2.0*n_trials)
    
plt.plot(c_values, fc_performance)
plt.plot(c_values, det_performance)
plt.xlabel('Coherence level')
plt.ylabel('Performance')

#%% Compare theoretical with forced choice performance
plt.plot(c_values,auc_performance,'-k')
plt.plot(c_values,fc_performance,'-ob')
plt.plot(c_values,det_performance,'-og')
plt.axhline(0.5,color='k',linestyle='--')
plt.xlabel('Coherence level')
plt.ylabel('Performance')
plt.legend(['AUC','2AFC','Threshold'],loc=4)