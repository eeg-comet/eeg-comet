
import numpy as np
from collections import Counter
from itertools import groupby


def fill_with_neighbors_with_higher_count(arr):
    """
    Fill the groups of -1 values in the array with the neighbor that has a higher count.
    """
    filled_arr = arr.copy()
    n = len(arr)
    i = 0

    while i < n:
        if arr[i] == -1:
            # Find the start and end indices of the group of -1 values
            start = i
            while i < n and arr[i] == -1:
                i += 1
            end = i - 1

            # Count the occurrences of the previous and next values
            prev = arr[start - 1] if start > 0 else 0
            next_val = arr[end + 1] if end < n - 1 else 0

            # Count the occurrences of the previous and next values in the windows
            prev_count = 0
            if start > 0:
                window_before = arr[max(start - 2, 0):start]
                prev_count = Counter(window_before).most_common(1)[0][1]

            next_count = 0
            if end < n - 1:
                window_after = arr[end + 2:min(end + 3, n)]
                next_count = Counter(window_after).most_common(1)[0][1]

            # Fill the group of -1 values with the neighbor with the higher count
            if prev_count > next_count:
                fill_value = prev
            elif prev_count < next_count:
                fill_value = next_val
            else:
                fill_value = prev or next_val

            # Fill the group with the determined fill value
            for j in range(start, end + 1):
                filled_arr[j] = fill_value
        else:
            i += 1

    return filled_arr


def fill_with_neighbors_half(arr):
    """
    Fill the groups of -1 values in the array by evenly distributing the neighboring values.
    """
    filled_arr = arr.copy()
    n = len(arr)
    i = 0
    count = 0

    while i < n:
        if arr[i] == -1:
            # Find the start and end indices of the group of -1 values
            start = i
            while i < n and arr[i] == -1:
                i += 1
                count += 1
            end = i - 1

            # Determine the fill values based on the previous and next elements
            fill_value_prev = arr[start - 1] if start > 0 else 0
            fill_value_next = arr[end + 1] if end < n - 1 else 0

            # Determine the number of elements for each fill value
            half_count = count // 2
            half_count_prev = half_count if count % 2 == 0 else half_count + 1
            half_count_next = half_count

            # Fill the group with the previous and next values
            for j in range(start, start + half_count_prev):
                filled_arr[j] = fill_value_prev
            for j in range(start + half_count_prev, end + 1):
                filled_arr[j] = fill_value_next

            count = 0
        else:
            i += 1

    return filled_arr

def segmentation_smooth(data, maps, n_states, epsilon=1e-6, b=3, lamb=5):
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

    print(epsilon, b, lamb)
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
    
    return segmentation

def substitude_maps_with_duration(segmentation, segments_less_than, option, data, maps, n_states, smooth_param=[]):
    """
    Substitute short segments in the segmentation array with neighboring elements based on the chosen option.
    """
    filled_segmentation = np.copy(segmentation)
    count_dups = [sum(1 for _ in group) for _, group in groupby(filled_segmentation)]

    # Replace short segments if left and right elements are the same
    for i, count in enumerate(count_dups):
        if count <= int(segments_less_than):
            start = int(np.sum(count_dups[0:i]))
            stop = int(start + count)
            # Check if there is a left and right neighbor to the segment
            if start > 0 and stop < len(filled_segmentation):
                # Check if the left and right neighbors are the same as the segment
                if filled_segmentation[start - 1] == filled_segmentation[stop]:
                    filled_segmentation[start:stop] = filled_segmentation[start - 1]

    # Recalculate count_dups after replacing segments with identical neighbors
    count_dups = [sum(1 for _ in group) for _, group in groupby(filled_segmentation)]

    # Find short segments
    for C in range(len(count_dups)):
        if count_dups[C] < int(segments_less_than):
            start = int(np.sum(count_dups[0:C]))
            stop = int(start + count_dups[C])
            filled_segmentation[start:stop] = -1

    filled_segmentation = np.array(filled_segmentation)

    if option == 'replace_high':
        # Replace short segments with nearby elements with higher occurrence
        filled_segmentation = fill_with_neighbors_with_higher_count(filled_segmentation)
    if option == 'replace_half':
        # Replace short segments by evenly distributing neighboring elements
        filled_segmentation = fill_with_neighbors_half(filled_segmentation)
    if option == 'smooth':
        # TODO: add user self define parameter option: epsilon=1e-6, b=3, lamb=5
        if not smooth_param:
            smooth_param = [1e-6, 3, 5]
        filled_segmentation = segmentation_smooth(data, maps, n_states, *smooth_param)

    return filled_segmentation
