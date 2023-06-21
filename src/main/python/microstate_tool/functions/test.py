# from modified_kmeans import modified_kmeans_table_2
import h5py
import pandas as pd
import numpy as np
from collections import Counter
# np.set_printoptions(threshold=100)

def modified_kmeans_table_2(data, maps, n_states, epsilon=1e-6, b=3, lamb=5):
	'''
	data: V
	maps: Gamma (T-like symbol)
	n_states: N_mu in paper
	epsilon: convergence criterion parameter
	b: window size parameter
	lamb: non-smoothness penalty parameter (lambda)
	
	segmentation: L
	n_channels: N_s in paper
	n_samples: N_T

	'''

	n_channels, n_samples = data.shape 
	data_sum_sq = np.sum(data ** 2)
	iteration = 0
	prev_residual = 0
	residual = np.inf
	thresh = epsilon

	# STEP 2 in TABLE 2
	# V dot Gamma
	activation = maps.dot(data)
	# L
	segmentation = np.argmax(np.abs(activation), axis=0) 
	print(f'SEG BEFORE: {segmentation[400:500]}')

	# STEP 3 in TABLE 2
	raw_segmentation = segmentation
	
	# STEP 4 in TABLE 2
	act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
	e1 = abs(data_sum_sq - act_sum_sq)
	e2 = e1 / float(n_samples * (n_channels - 1)) 

	while residual > thresh:
		print(f'iteration: {iteration+1} residual: {residual}, thresh: {thresh}')
		
		# STEP 5 in TABLE 2
		windows = np.lib.stride_tricks.sliding_window_view(raw_segmentation, 2*b+1)
		N_bkt = np.zeros((windows.shape[0], n_states))
		for i, window in enumerate(windows):
			cnt = Counter(window)
			N_bkt[i] = [cnt[x] for x in range(n_states)]
		raw_segmentation[b:n_samples-b] = np.argmin((np.sum(data**2, axis=0) - (np.sum(maps[segmentation].T * data, axis=0) ** 2))[b:n_samples-b] / (2*e2*(n_channels - 1)) - (lamb*N_bkt).T, axis=0)

		# STEP 6 in TABLE 2
		segmentation = raw_segmentation# .copy()
		
		# STEP 7 in TABLE 2
		act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
		e1 = abs(data_sum_sq - act_sum_sq)
		sigma_mu = (e1 / float(n_samples * (n_channels - 1)))
		residual = abs(prev_residual - sigma_mu)

		# STEP 8 in TABLE 2
		prev_residual = sigma_mu
		thresh = epsilon * sigma_mu
		iteration += 1

	print(f'SEG AFTER : {segmentation[400:500]}')
	print('Finishes after', str(iteration), 'Iterations.')
	
	# STEP 9 in TABLE 2
	# WARN: seems like we didn't use the result of this step
	
	# STEP 10 in TABLE 2
	# sigma_d_squared = data_sum_sq / float(n_samples * (n_channels - 1))
	# R_squared = 1 - sigma_mu/sigma_d_squared
	
	# return segmentation

def main():
	# TODO: remember to change the path of the testing files
	data = h5py.File('/Users/bottlecap/Downloads/output/t2/preprocessed_data/sub-032302_EC.hdf', 'r')
	data = data[list(data.keys())[0]]
	data = np.asarray(data)
	maps = pd.read_csv('/Users/bottlecap/Downloads/output/t2/raw_features/microstate_maps.csv').iloc[:, 1:].values.T
	maps /= np.linalg.norm(maps)
	
	modified_kmeans_table_2(data, maps, n_states=5, epsilon=1e-6, b=3, lamb=5)



if __name__ == '__main__':
	main()
