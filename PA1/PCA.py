################################################################################
# CSE 251B: Programming Assignment 1
# Code snippet by Divyanshu
# Winter 2021
################################################################################
# We've provided you with the dataset in PA1.zip
################################################################################

import numpy as np

def PCA(X, n_components):
	"""
	Args:
		X: has shape Mxd where M is the number of images and d is the dimension of each image
		n_components: The number of components you want to project your image onto. 
	
	Returns:
		projected: projected data of shape M x n_components
		mean_image: mean of all images
		top_sqrt_eigen_values: singular values
		top_eigen_vectors: eigenvectors 
	"""

	mean_image = np.average(X, axis = 0)

	msd = X - mean_image # M x d

	smart_cov_matrix = np.matmul(msd, msd.T)
	eigen_values, smart_eigen_vectors = np.linalg.eig(smart_cov_matrix)

	idx = eigen_values.argsort()[::-1]   
	eigen_values = eigen_values[idx]
	smart_eigen_vectors = smart_eigen_vectors[:,idx]

	eigen_vectors = (np.matmul(msd.T, smart_eigen_vectors)).T # M x d

	row_norm = np.sum(np.abs(eigen_vectors)**2,axis=-1)**(1./2) # M

	normalized_eigen_vectors = eigen_vectors/(row_norm.reshape(-1, 1)) # M x d

	top_eigen_vectors = normalized_eigen_vectors[:n_components].T
	top_sqrt_eigen_values = np.sqrt(eigen_values[:n_components])

	projected = np.matmul(msd, top_eigen_vectors)/top_sqrt_eigen_values

	return projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors