# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:27:58 2017

@author: Christophe
"""

import numpy as np
import PIL
import matplotlib.pyplot as plt

# Specify the folder where the face images are downloaded and unzipped
# Database can be downloaded from the following site:
# http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
face_dir = r'C:\Users\Christophe\Documents\GitHub\Computational-Neuroscience-UW\week-02\orl_faces'


# Load the images. To save memory the image is rescaled to 50 by 60 pixels
# Only the first image in each folder is loaded
n_faces   = 40 
face_data = np.zeros((3000,n_faces))
for i in range(1, n_faces + 1):
    subject_folder = 's' + str(i)
    pil_image = PIL.Image.open(face_dir + '\\' + subject_folder + '\\1.pgm').resize((50,60))
    face_data[:,i-1] = np.array(pil_image).flatten()

# Perform standard pre-processing operations  
mean_face = np.mean(face_data,1)
sd_face   = np.std(face_data,1)
face_data = face_data - mean_face[:,np.newaxis]
face_data = face_data/sd_face[:,np.newaxis]
plt.imshow(np.reshape(face_data[:,0],(60,50)),cmap = 'gray')

# Eigenfaces are computed by calculating the eigenvalues
# of the pixel covariance matrix for all faces
covariance_matrix = np.cov(face_data)
[w,v] = np.linalg.eig(covariance_matrix)


# Find the indices of the largest eigenvalues and show the associated
# eigenvectors. 
l_indices = np.argsort(w)
eigenface_1 = np.real(np.reshape(v[:,l_indices[-1]], (60,50)))
eigenface_2 = np.real(np.reshape(v[:,l_indices[-2]], (60,50)))

plt.subplot(1,2,1)
plt.imshow(eigenface_1, cmap= 'gray')
plt.axis('off')
plt.title('Eigenface #1')
plt.subplot(1,2,2)
plt.imshow(eigenface_2, cmap = 'gray')
plt.axis('off')
plt.title('Eigenface #2')

# Project face on eigenvectors to get component loading and show 
# reconstructed images
n_eigen_faces = [5, 10, 40]
face_reconstruction = np.zeros((3,3000))

for f in range(3):
    for i in range(n_eigen_faces[f]):
        loading = np.dot(face_data[:,0], v[:,l_indices[-1-i]])
        face_reconstruction[f,:] += np.real(loading*v[:,l_indices[-1-i]])
    
plt.subplot(2,3,2)    
plt.imshow(np.reshape(face_data[:,0], (60,50)),cmap='gray')
plt.axis('off')
plt.title('Original image')
plt.subplot(2,3,4)
plt.imshow(np.reshape(face_reconstruction[0,:], (60,50)),cmap='gray')
plt.axis('off')
plt.title('EF = 5')
plt.subplot(2,3,5)
plt.imshow(np.reshape(face_reconstruction[1,:], (60,50)),cmap='gray')
plt.axis('off')
plt.title('EF = 10')
plt.subplot(2,3,6)
plt.imshow(np.reshape(face_reconstruction[2,:], (60,50)),cmap='gray')
plt.axis('off')
plt.title('EF = 40')