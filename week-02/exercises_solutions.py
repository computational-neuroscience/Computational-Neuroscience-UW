# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 16:59:15 2016

@author: Christophe
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
T = 10
r = 100

p = r*dt

spikes = np.random.rand(int(T/dt))
spikes[spikes < p] = 1
spikes[spikes < 1] = 0

isi = np.diff(np.where(spikes == 1))

# Coefficient of variation = standard deviation/mean
cv = np.std(isi)/np.mean(isi)
print('Coefficient of variation: ' + str(cv))

# Fano factor
