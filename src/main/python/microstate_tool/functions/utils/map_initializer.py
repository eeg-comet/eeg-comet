
import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def map_initializer(data, maps, peaks, n_states, initializer):
    # create instance of K-Means algorithm with prepared centers
    if initializer == 'Random':
        random_state = np.random.RandomState(None)
        chosen_peaks = random_state.choice(len(peaks), size=n_states, replace=False)
        initial_peaks = peaks[chosen_peaks].tolist()
        initial_centers = data[:, initial_peaks].T
    elif initializer == 'K-Means++':
        # Calculate initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(maps, n_states).initialize()
    initial_centers /= np.linalg.norm(initial_centers, axis=1, keepdims=True)
    return initial_centers
