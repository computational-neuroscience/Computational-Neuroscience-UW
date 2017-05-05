# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:59:45 2017

@author: Christophe

This script was used to investigate a question on Coursera about the likelihood
ratio test. The question was if a test involving the ratio of areas (or cummulative 
distribution) would be more appropriate. In the first part I simulate the likelihood
ratio and the area ratio to show that they produce different response criteria.

In the second part I draw samples from each distribution, and show that compared
to the area ratio test, the likelihood ratio test produces the best results in terms
of total error.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Parameters of the Gaussian distributions
mu_1 = 0
mu_2 = 4

sigma_1 = 1.0
sigma_2 = 4.0

# Compute the probability density function
x = np.arange(-4,10,0.01)
pdf_1 = norm.pdf(x,mu_1,sigma_1)
pdf_2 = norm.pdf(x,mu_2,sigma_2)

cdf_1 = norm.cdf(x,mu_1,sigma_1)
cdf_2 = norm.cdf(x,mu_2,sigma_2)

# Compute the two likelihood ratio tests
lr_1 = np.log(pdf_2/pdf_1)
lr_2 = np.log((cdf_2)/(1-cdf_1))

plt.subplot(2,2,1)
plt.plot(x,pdf_1)
plt.plot(x,pdf_2)
plt.legend(['P(r|-)','P(r|+)'])
plt.title('Probability density')

plt.subplot(2,2,2)
plt.plot(x,1-cdf_1)
plt.plot(x,cdf_2)
plt.title('Cummulative distribution')
plt.legend(['1-P(x<r|-)','P(x<r|+)'])

plt.subplot(2,1,2)
plt.plot(x,lr_1)
plt.plot(x,lr_2)
plt.axhline(0,color='k',linewidth=0.5)
plt.title('Ratio tests')
plt.legend(['pdf ratio','cdf ratio'])
plt.ylim(-5,20)

# Get the location where ratio exceeds threshold. This is just a quick way
# and probably does not work in all cases
eps = 0.01
np.where((lr_1 < eps) & (lr_1 > -eps))[0]
np.where((lr_2 < eps) & (lr_2 > -eps))[0]
x1 = 171
x2 = 576
x3 = 480
plt.plot(x[x1],lr_1[x1],'og')
plt.plot(x[x2],lr_1[x2],'og')
plt.plot(x[x3],lr_2[x3],'or')
#%%
n_samples = 1000
samples_1 = np.random.normal(mu_1,sigma_1,n_samples)
samples_2 = np.random.normal(mu_2,sigma_2,n_samples)


# Calculate the standard likelihood ratio test for data from each sample
lr_1 = norm.pdf(samples_1,mu_2,sigma_2)/norm.pdf(samples_1,mu_1,sigma_1)
lr_2 = norm.pdf(samples_2,mu_2,sigma_2)/norm.pdf(samples_2,mu_1,sigma_1)

# Summmary statistics for the standard likelihood ratio test
hits         = len(np.where(lr_2 > 1)[0])
miss         = len(np.where(lr_2 < 1)[0])
false_alarm  = len(np.where(lr_1 > 1)[0])
correct_rej  = len(np.where(lr_1 < 1)[0])
print('Likelihood ratio test statistics: ')
print('Hits: ' + str(hits) + ', Correct rejections: ' + str(correct_rej))
print('Miss: ' + str(miss) + ', False alarms      : ' + str(false_alarm))
print('Percentage correct: ' + str((hits+correct_rej)/(2*n_samples)))
print('\n')

# Calculate the ratio test involving areas
lr_1 = norm.cdf(samples_1,mu_2,sigma_2)/(1-norm.cdf(samples_1,mu_1,sigma_1))
lr_2 = norm.cdf(samples_2,mu_2,sigma_2)/(1-norm.cdf(samples_2,mu_1,sigma_1))

# Summary statistics for the area ratio test
hits         = len(np.where(lr_2 >= 1)[0])
miss         = len(np.where(lr_2 < 1)[0])
false_alarm  = len(np.where(lr_1 >= 1)[0])
correct_rej  = len(np.where(lr_1 < 1)[0])
print('Area ratio test statistics: ')
print('Hits: ' + str(hits) + ', Correct rejections: ' + str(correct_rej))
print('Miss: ' + str(miss) + ', False alarms      : ' + str(false_alarm))
print('Percentage correct: ' + str((hits+correct_rej)/(2*n_samples)))
print('\n')