import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class MicrostateClusterer:
    """A class for clustering EEG data into microstates using various algorithms.

    This class implements several microstate clustering algorithms including Modified K-means,
    Modified K-means with similarity metrics, and Topographic Atomize and Agglomerate
    Hierarchical Clustering (TAAHC). These algorithms are specifically designed for EEG
    topography analysis.

    Attributes:
        n_states (int): Number of microstate maps to identify.
        batch_size (int, optional): Number of samples to process at once for memory optimization.
        number_of_repeats (int): Number of clustering initializations to try.
        max_iterations (int): Maximum number of iterations per clustering attempt.
        clustering_tolerance (float): Convergence tolerance threshold.
        best_maps (ndarray, optional): Best microstate maps found during clustering.
    """

    def __init__(self, n_states, batch_size=None, n_inits=10, max_iter=500, tolerance=1e-6):
        """Initialize the MicrostateClusterer with clustering parameters.

        Args:
            n_states (int): Number of microstate maps to identify.
            batch_size (int, optional): Number of samples to process at once for memory optimization.
                Default is None (process all data at once).
            n_inits (int, optional): Number of clustering initializations to try. Default is 10.
            max_iter (int, optional): Maximum number of iterations per clustering attempt. Default is 500.
            tolerance (float, optional): Convergence tolerance threshold. Default is 1e-6.
        """
        self.n_states = n_states
        self.batch_size = batch_size
        self.number_of_repeats = n_inits
        self.max_iterations = max_iter
        self.clustering_tolerance = tolerance
        self.best_maps = None

    # --------------------------------------------------------------------------
    # CORE CLUSTERING ALGORITHMS
    # --------------------------------------------------------------------------

    def modified_kmeans(self, data, initial_maps, verbose=True, worker=None):
        """Perform topographic clustering of EEG data to identify brain microstates.

        This implements the modified K-means clustering algorithm described by
        Pascual-Marqui, Michel, and Lehmann (1995). Unlike standard K-means,
        this algorithm:

        1. Assigns clusters based on maximum correlation (ignoring polarity)
        2. Updates maps by weighted averaging based on activations
        3. Enforces map normalization after each update
        4. Optimizes for explained variance

        Args:
            data (ndarray): Input EEG data with shape (n_channels, n_samples).
            initial_maps (ndarray): Initial microstate maps with shape (n_states, n_channels).
            verbose (bool, optional): Whether to print iteration information. Default is True.

        Returns:
            tuple:
                maps (ndarray): Final microstate maps with shape (n_states, n_channels).
                residual (float): Final residual variance not explained by the model.

        References:
            Pascual-Marqui, R.D., Michel, C.M., Lehmann, D. (1995). Segmentation of
            brain electrical activity into microstates: model estimation and validation.
            IEEE Transactions on Biomedical Engineering, 42(7), 658-665.
        """
        # Initial setup
        n_channels, n_samples = data.shape
        maps = initial_maps.copy()
        data_sum_sq = np.sum(data ** 2)
        prev_residual = np.inf

        # Determine if we use batch processing
        use_batches = self.batch_size is not None and self.batch_size > 0

        if use_batches and verbose:
            print(f"[CLUSTERING] Using batch processing with batch size: {self.batch_size}")
            batch_count = int(np.ceil(n_samples / self.batch_size))

        # Clustering iterations
        for iteration in range(self.max_iterations):
            # Check if we should stop
            if worker and hasattr(worker, 'stopped') and worker.stopped:
                if verbose:
                    print(f"\n[CLUSTERING] Clustering stopped at iteration {iteration}")
                return maps, prev_residual  # Return current best maps

            # Initialize arrays for segmentation and activations
            segmentation = np.zeros(n_samples, dtype=int)

            if use_batches:
                # Process data in batches
                map_sums = np.zeros((self.n_states, n_channels))
                act_sum_sq = 0

                for b, batch_start in enumerate(range(0, n_samples, self.batch_size)):
                    batch_end = min(batch_start + self.batch_size, n_samples)
                    batch_data = data[:, batch_start:batch_end]
                    batch_size_actual = batch_end - batch_start

                    if verbose and iteration == 0 and (b % 10 == 0 or b == batch_count - 1):
                        print(f"[CLUSTERING] Processing batch {b + 1}/{batch_count} (samples {batch_start}-{batch_end})")

                    # Assign each sample in the batch to the best matching microstate
                    batch_activation = maps.dot(batch_data)
                    batch_segmentation = np.argmax(np.abs(batch_activation), axis=0)
                    
                    # Validate segmentation indices
                    if np.any(batch_segmentation >= self.n_states) or np.any(batch_segmentation < 0):
                        print(f"⚠️  [CLUSTERING] Invalid batch segmentation indices. Max: {np.max(batch_segmentation)}, Min: {np.min(batch_segmentation)}, n_states: {self.n_states}")
                        # Clip indices to valid range
                        batch_segmentation = np.clip(batch_segmentation, 0, self.n_states - 1)

                    # Store segmentation for this batch
                    segmentation[batch_start:batch_end] = batch_segmentation

                    # Accumulate map sums for later updates
                    for state in range(self.n_states):
                        idx = (batch_segmentation == state)
                        if np.sum(idx) > 0:
                            map_sums[state] += np.dot(batch_data[:, idx], batch_activation[state, idx])

                    # Accumulate activation sum squared for residual calculation
                    batch_act_sum_sq = np.sum(np.sum(maps[batch_segmentation].T * batch_data, axis=0) ** 2)
                    act_sum_sq += batch_act_sum_sq

                # Update maps after processing all batches
                for state in range(self.n_states):
                    if np.linalg.norm(map_sums[state]) > 0:
                        maps[state] = map_sums[state]
                        maps[state] /= np.linalg.norm(maps[state])

            else:
                # Process all data at once (original implementation)
                activation = maps.dot(data)
                segmentation = np.argmax(np.abs(activation), axis=0)
                
                # Validate segmentation indices
                if np.any(segmentation >= self.n_states) or np.any(segmentation < 0):
                    print(f"⚠️  [CLUSTERING] Invalid segmentation indices in non-batch mode. Max: {np.max(segmentation)}, Min: {np.min(segmentation)}, n_states: {self.n_states}")
                    # Clip indices to valid range
                    segmentation = np.clip(segmentation, 0, self.n_states - 1)

                # Update maps
                for state in range(self.n_states):
                    idx = (segmentation == state)
                    if np.sum(idx) > 0:
                        maps[state] = np.dot(data[:, idx], activation[state, idx])
                        maps[state] /= np.linalg.norm(maps[state])

                # Calculate activation sum squared
                act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)

            # Estimate residual noise
            residual = abs(data_sum_sq - act_sum_sq) / float(n_samples * (n_channels - 1))

            # Check for convergence
            if (prev_residual - residual) < (self.clustering_tolerance * residual):
                if verbose:
                    print(f'[CLUSTERING] Clustering converged at {iteration} iterations')
                break

            prev_residual = residual

        return maps, residual

    def modified_kmeans_similarity(self, data, initial_maps, metric='Cosine Similarity', verbose=True, worker=None):
        """Perform K-means clustering using spatial similarity metrics.

        This variant of the modified K-means algorithm uses either cosine similarity
        or spatial correlation to assign samples to clusters, which better handles
        the polarity-independent nature of EEG topographies.

        Args:
            data (ndarray): Input EEG data with shape (n_channels, n_samples).
            initial_maps (ndarray): Initial microstate maps with shape (n_states, n_channels).
            metric (str, optional): Similarity metric to use ('Cosine Similarity' or
                'Spatial Correlation'). Default is 'Cosine Similarity'.
            verbose (bool, optional): Whether to print iteration information. Default is True.

        Returns:
            tuple:
                maps (ndarray): Final microstate maps with shape (n_states, n_channels).
                residual (float): Final residual variance not explained by the model.

        Raises:
            ValueError: If an invalid similarity metric is specified.

        Notes:
            This approach tends to ignore polarity differences, making it suitable
            for identifying distinct spatial patterns regardless of field orientation.
        """
        # Initial setup
        n_channels, n_samples = data.shape
        maps = initial_maps.copy()
        prev_residual = np.inf

        # Validate similarity metric
        if metric not in ['Cosine Similarity', 'Spatial Correlation']:
            raise ValueError(
                "Invalid similarity metric. Valid options are 'Cosine Similarity' and 'Spatial Correlation'.")

        # Determine if we use batch processing
        use_batches = self.batch_size is not None and self.batch_size > 0

        if use_batches and verbose:
            print(f"[CLUSTERING] Using batch processing with batch size: {self.batch_size}")
            print(f"[CLUSTERING] Similarity metric: {metric}")
            batch_count = int(np.ceil(n_samples / self.batch_size))

        # Clustering iterations
        for iteration in range(self.max_iterations):
            # Check if we should stop
            if worker and hasattr(worker, 'stopped') and worker.stopped:
                if verbose:
                    print(f"\n[CLUSTERING] Clustering stopped at iteration {iteration}")
                return maps, prev_residual  # Return current best maps

            # Initialize arrays for segmentation and best similarities
            segmentation = np.zeros(n_samples, dtype=int)
            best_similarities = np.zeros(n_samples)

            if use_batches:
                # Process data in batches
                map_updates = np.zeros((self.n_states, n_channels))
                map_weights = np.zeros(self.n_states)

                for b, batch_start in enumerate(range(0, n_samples, self.batch_size)):
                    batch_end = min(batch_start + self.batch_size, n_samples)
                    batch_data = data[:, batch_start:batch_end]
                    batch_size_actual = batch_end - batch_start

                    if verbose and iteration == 0 and (b % 10 == 0 or b == batch_count - 1):
                        print(f"[CLUSTERING] Processing batch {b + 1}/{batch_count} (samples {batch_start}-{batch_end})")

                    # Calculate similarities between maps and data samples
                    if metric == 'Cosine Similarity':
                        # Normalize maps and data for cosine similarity
                        maps_norm = maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-10)
                        batch_norm = batch_data / (np.linalg.norm(batch_data, axis=0, keepdims=True) + 1e-10)
                        similarities = np.abs(np.dot(maps_norm, batch_norm))
                    else:  # Spatial Correlation
                        # Normalize maps for correlation
                        maps_norm = (maps - maps.mean(axis=1, keepdims=True)) / (
                                    maps.std(axis=1, keepdims=True) + 1e-10)
                        batch_norm = (batch_data - batch_data.mean(axis=0)) / (batch_data.std(axis=0, ddof=1) + 1e-10)
                        similarities = np.abs(np.dot(maps_norm, batch_norm))

                    # Assign each sample to the best matching microstate
                    batch_segmentation = np.argmax(similarities, axis=0)
                    batch_best_similarities = np.max(similarities, axis=0)
                    
                    # Validate segmentation indices
                    if np.any(batch_segmentation >= self.n_states) or np.any(batch_segmentation < 0):
                        print(f"⚠️  [CLUSTERING] Invalid batch segmentation indices in similarity mode. Max: {np.max(batch_segmentation)}, Min: {np.min(batch_segmentation)}, n_states: {self.n_states}")
                        # Clip indices to valid range
                        batch_segmentation = np.clip(batch_segmentation, 0, self.n_states - 1)

                    # Store segmentation and best similarities for this batch
                    segmentation[batch_start:batch_end] = batch_segmentation
                    best_similarities[batch_start:batch_end] = batch_best_similarities

                    # Accumulate weighted data for map updates
                    for state in range(self.n_states):
                        idx = (batch_segmentation == state)
                        if np.sum(idx) > 0:
                            map_updates[state] += np.dot(batch_data[:, idx], similarities[state, idx])
                            map_weights[state] += np.sum(similarities[state, idx])

                # Update maps after processing all batches
                for state in range(self.n_states):
                    if map_weights[state] > 0:
                        maps[state] = map_updates[state] / map_weights[state]
                        maps[state] /= np.linalg.norm(maps[state])

            else:
                # Process all data at once (original implementation)
                # Calculate distances/similarities between maps and data
                if metric == 'Cosine Similarity':
                    distances = cdist(maps, data.T, 'cosine')
                else:  # Spatial Correlation
                    distances = cdist(maps, data.T, 'correlation')

                # Convert distances to similarities (1 - distance)
                similarities = 1 - np.abs(distances)

                # Assign each sample to the best matching microstate
                segmentation = np.argmax(similarities, axis=0)
                best_similarities = np.max(similarities, axis=0)

                # Update maps
                for state in range(self.n_states):
                    idx = (segmentation == state)
                    if np.sum(idx) > 0:
                        maps[state] = np.dot(data[:, idx], similarities[state, idx])
                        maps[state] /= np.linalg.norm(maps[state])

            # Calculate residual (1 - average of best similarities)
            residual = 1 - np.mean(best_similarities)

            # Check for convergence
            if (prev_residual - residual) < (self.clustering_tolerance * residual):
                if verbose:
                    print(f'[CLUSTERING] Clustering converged at {iteration} iterations')
                break

            prev_residual = residual

        return maps, residual

    def taahc(self, data, metric='Spatial Correlation', verbose=True, progress_callback=None, worker=None):
        """Perform Topographic Atomize and Agglomerate Hierarchical Clustering (TAAHC) for EEG microstates.

        This memory-optimized implementation of the TAAHC algorithm clusters EEG data
        by iteratively removing the weakest microstate and reassigning its samples.

        Args:
            data (ndarray): Input EEG data with shape (n_channels, n_samples).
            metric (str, optional): Similarity metric to use ('Cosine Similarity' or
                'Spatial Correlation'). Default is 'Spatial Correlation'.
            verbose (bool, optional): Whether to print progress information. Default is True.
            progress_callback (callable, optional): Callback function for progress updates.
                Should accept (current, total, message). Added for GUI integration.

        Returns:
            tuple:
                maps (ndarray): Final microstate maps with shape (n_states, n_channels).
                residual (float): Final residual variance not explained by the model.

        Raises:
            ValueError: If an invalid similarity metric is specified.

        Notes:
            This algorithm starts with each GFP peak as a potential microstate and
            iteratively merges states until the target number is reached. It's suitable
            for datasets where the number of GFP peaks is greater than the desired
            number of microstates.
        """
        import time

        if verbose:
            print(f"[CLUSTERING] Starting TAAHC clustering")
            print(f"[CLUSTERING] Using similarity metric: {metric}")
            start_time = time.time()

        # Validate similarity metric
        if metric not in ['Cosine Similarity', 'Spatial Correlation']:
            raise ValueError(
                "Invalid similarity metric. Valid options are 'Cosine Similarity' and 'Spatial Correlation'.")

        n_channels, n_samples = data.shape

        # Set default batch size if None
        if self.batch_size is None or self.batch_size <= 0:
            # Use a reasonable default batch size based on data size
            # For large datasets, use smaller batches to manage memory
            if n_samples > 100000:
                batch_size = 10000
            elif n_samples > 50000:
                batch_size = 5000
            else:
                batch_size = min(1000, n_samples)
        else:
            batch_size = self.batch_size

        if verbose:
            print(f"[CLUSTERING] Using batch size: {batch_size}")

        # Initialize progress tracking with better estimation
        def update_progress(message, step_increment=1):
            nonlocal current_step
            current_step += step_increment
            if progress_callback and total_estimated_steps > 0:
                progress_callback(current_step, total_estimated_steps, message)

        # Initialize progress tracking variables
        current_step = 0
        total_estimated_steps = 100  # Initial placeholder, will be updated after peaks are found
        total_iterations_needed = 0  # Will be updated after peaks are found
        
        # Progress tracking variables
        iteration_times = []  # Track time per iteration for ETA calculation
        last_eta_update = 0  # Track when we last updated ETA

        # Step 1: Calculate GFP (Global Field Power)
        update_progress("Calculating GFP curve...")
        gfp_curve = np.std(data, axis=0)

        # Step 2: Find GFP peaks (local maxima)
        update_progress("Detecting GFP peaks...")
        peaks = np.where((gfp_curve[:-2] < gfp_curve[1:-1]) &
                         (gfp_curve[1:-1] > gfp_curve[2:]))[0] + 1

        if verbose:
            print(f"[CLUSTERING] Found {len(peaks)} GFP peaks")

        if len(peaks) < self.n_states:
            update_progress(f"Adding random samples (found only {len(peaks)} peaks)...")
            if verbose:
                print(f"⚠️  [CLUSTERING] Only {len(peaks)} GFP peaks found, less than requested {self.n_states} states")
                print("[CLUSTERING] Adding random samples to reach required number of initial states")
            # Add random samples if needed
            additional = np.random.choice(np.arange(n_samples),
                                          size=max(self.n_states - len(peaks), 0),
                                          replace=False)
            peaks = np.concatenate([peaks, additional])
            if verbose:
                print(f"[CLUSTERING] Added {len(additional)} random samples as initial states")

        # Step 3: Initialize with peak maps
        update_progress("Initializing with peak maps...")
        peak_data = data[:, peaks]
        maps = peak_data.T.copy()  # Shape: (n_peaks, n_channels)
        n_maps = maps.shape[0]
        n_peaks = len(peaks)  # Define n_peaks for use in progress tracking
        
        # Update progress estimation with actual number of peaks
        total_iterations_needed = max(n_peaks - self.n_states, 0)
        total_estimated_steps = (
                15 +  # Initial setup steps (GFP, peaks, initialization)
                total_iterations_needed * 8 +  # Each iteration: batch processing + atomization + reassignment + recalculation
                25  # Final processing steps (residual calculation)
        )

        # Normalize maps
        for i in range(n_maps):
            maps[i] /= np.linalg.norm(maps[i])

        # Track cluster indices
        cluster_indices = [[k] for k in range(n_maps)]

        # For GEV calculation
        data_sum_sq = np.sum(gfp_curve ** 2)

        if verbose:
            print(f"[CLUSTERING] Starting hierarchical clustering with {n_maps} maps")
            print("=" * 70)

        update_progress(f"Starting hierarchical clustering: {n_maps} → {self.n_states} maps...")

        iteration = 0
        iteration_start_time = time.time()
        
        # Main TAAHC loop
        while n_maps > self.n_states:
            # Check if we should stop
            if worker and hasattr(worker, 'stopped') and worker.stopped:
                if verbose:
                    print(f"\n[CLUSTERING] Clustering stopped by user with {n_maps} maps remaining")
                    print(f"[CLUSTERING] Cannot save maps: target {self.n_states} states not reached (current: {n_maps})")
                # Return None for maps since we don't have the correct number of states
                return None, np.inf

            # Check progress callback return value
            if progress_callback:
                continue_processing = progress_callback(
                    current_step,
                    total_estimated_steps,
                    f"TAAHC: {n_maps} → {self.n_states} maps (iteration {iteration + 1})"
                )
                if not continue_processing:
                    if verbose:
                        print(f"\n[CLUSTERING] Clustering stopped by progress callback")
                        print(f"[CLUSTERING] Cannot save maps: target {self.n_states} states not reached (current: {n_maps})")
                    # Return None for maps since we don't have the correct number of states
                    return None, np.inf

            iteration += 1
            iteration_iter_start = time.time()

            # Update progress with current iteration info
            remaining_iterations = n_maps - self.n_states
            progress_msg = f"TAAHC Iteration {iteration}: {n_maps} → {n_maps - 1} maps ({remaining_iterations} remaining)"
            update_progress(progress_msg)

            # Enhanced logging with ETA every 10th iteration
            if verbose and (iteration % 10 == 0 or n_maps <= self.n_states + 5 or iteration == 1):
                elapsed = time.time() - start_time
                
                # Calculate ETA if we have enough data
                eta_msg = ""
                if len(iteration_times) >= 3:
                    avg_iter_time = np.mean(iteration_times[-10:])  # Use last 10 iterations
                    eta_seconds = avg_iter_time * remaining_iterations
                    eta_minutes = eta_seconds / 60
                    eta_msg = f", ETA: {eta_minutes:.1f} min"
                
                print(f"[CLUSTERING] TAAHC Iteration {iteration}: {n_maps} maps remaining ({remaining_iterations} to go), "
                      f"elapsed: {elapsed:.1f}s{eta_msg}")

            # Initialize arrays to store assignments and best correlations
            assignments = np.zeros(n_samples, dtype=int)
            best_corrs = np.zeros(n_samples)

            # Normalize maps based on the chosen metric
            if metric == 'Spatial Correlation':
                # Normalize for correlation (subtract mean, divide by std)
                maps_norm = (maps - maps.mean(axis=1, keepdims=True)) / (maps.std(axis=1, keepdims=True) + 1e-10)
            else:  # Cosine Similarity
                # Normalize for cosine similarity (just divide by norm)
                maps_norm = maps / np.linalg.norm(maps, axis=1, keepdims=True)

            # Process data in batches
            if verbose and n_maps <= self.n_states + 10:
                print(f"[CLUSTERING] Processing {n_samples} samples in batches of {batch_size}...")
                batch_count = int(np.ceil(n_samples / batch_size))

            for b, batch_start in enumerate(range(0, n_samples, batch_size)):
                batch_end = min(batch_start + batch_size, n_samples)

                batch_data = data[:, batch_start:batch_end]

                # Normalize batch data based on the chosen metric
                if metric == 'Spatial Correlation':
                    # Normalize for correlation (subtract mean, divide by std)
                    batch_norm = (batch_data - batch_data.mean(axis=0)) / (batch_data.std(axis=0, ddof=1) + 1e-10)
                else:  # Cosine Similarity
                    # Normalize for cosine similarity (just divide by norm)
                    batch_norm = batch_data / (np.linalg.norm(batch_data, axis=0, keepdims=True) + 1e-10)

                # Calculate similarities for this batch
                batch_corrs = np.abs(np.dot(maps_norm, batch_norm))

                # Store best assignments and correlations
                batch_assignments = np.argmax(batch_corrs, axis=0)
                batch_best_corrs = np.max(batch_corrs, axis=0)

                assignments[batch_start:batch_end] = batch_assignments
                best_corrs[batch_start:batch_end] = batch_best_corrs

            if verbose and n_maps <= self.n_states + 10:
                print("[CLUSTERING] Calculating atomization values for each map...")

            # Calculate atomization criterion for each map
            atomisation_values = np.zeros(n_maps)
            cluster_sizes = np.zeros(n_maps, dtype=int)

            for k in range(n_maps):
                mask = assignments == k
                cluster_sizes[k] = np.sum(mask)

                if cluster_sizes[k] == 0:  # Empty cluster
                    atomisation_values[k] = 0
                    continue

                # Correlation-based criterion
                atomisation_values[k] = np.sum(best_corrs[mask] ** 2)

            # Find worst map to remove
            worst_idx = np.argmin(atomisation_values)

            if verbose and n_maps <= self.n_states + 10:
                print(f"[CLUSTERING] Removing map #{worst_idx} with TAAHC value: {atomisation_values[worst_idx]:.6f}")

            # Update progress for map removal
            update_progress(f"Removing weakest map (#{worst_idx}), reassigning {cluster_sizes[worst_idx]} points...")

            # Remove the worst map
            maps = np.delete(maps, worst_idx, axis=0)

            # Get and remove indices from the worst cluster
            removed_indices = cluster_indices.pop(worst_idx)

            # Get data points from removed cluster
            removed_data = peak_data[:, removed_indices].T

            # Normalize based on the chosen metric
            if metric == 'Spatial Correlation':
                # Normalize for correlation
                removed_norm = (removed_data - removed_data.mean(axis=1, keepdims=True)) / (
                        removed_data.std(axis=1, keepdims=True) + 1e-10)
                maps_norm = (maps - maps.mean(axis=1, keepdims=True)) / (maps.std(axis=1, keepdims=True) + 1e-10)
            else:  # Cosine Similarity
                # Normalize for cosine similarity
                removed_norm = removed_data / (np.linalg.norm(removed_data, axis=1, keepdims=True) + 1e-10)
                maps_norm = maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-10)

            # Calculate similarity with remaining maps
            reassign_corr = np.abs(np.dot(removed_norm, maps_norm.T))
            reassign_idx = np.argmax(reassign_corr, axis=1)

            # Track which clusters were updated
            updated_clusters = set()

            # Reassign points
            for i, idx in enumerate(reassign_idx):
                cluster_indices[idx].append(removed_indices[i])
                updated_clusters.add(idx)

            # Update progress for cluster center recalculation
            update_progress(f"Recalculating {len(updated_clusters)} cluster centers...")

            # Recalculate centers for updated clusters
            for idx in updated_clusters:
                cluster_data = peak_data[:, cluster_indices[idx]].T

                # Use first principal component as the new map
                if cluster_data.shape[0] > 1:
                    # Memory efficient PCA - avoid full covariance matrix if possible
                    if cluster_data.shape[0] <= cluster_data.shape[1]:  # More features than samples
                        cov_matrix = np.dot(cluster_data, cluster_data.T)
                        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
                        # Get principal component
                        pc = np.dot(cluster_data.T, eigvecs[:, -1])
                    else:
                        cov_matrix = np.dot(cluster_data.T, cluster_data)
                        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
                        pc = eigvecs[:, -1]

                    maps[idx] = pc / np.linalg.norm(pc)
                else:
                    maps[idx] = cluster_data[0] / np.linalg.norm(cluster_data[0])

            n_maps = len(cluster_indices)
            
            # Track iteration time for ETA calculation
            iteration_time = time.time() - iteration_iter_start
            iteration_times.append(iteration_time)

            if verbose and n_maps == self.n_states:
                print("=" * 70)
                print(f"[CLUSTERING] Reached target of {self.n_states} states after {iteration} iterations")
                total_time = time.time() - start_time
                avg_iter_time = np.mean(iteration_times) if iteration_times else 0
                print(f"[CLUSTERING] Total time: {total_time:.1f}s, Average iteration time: {avg_iter_time:.3f}s")

        # Final processing
        update_progress("Calculating final assignments and residual...")
        if verbose:
            print("[CLUSTERING] Calculating final assignments and residual...")

        # Calculate final residual - using batches
        final_corrs = np.zeros(n_samples)

        # Final normalization based on the chosen metric
        if metric == 'Spatial Correlation':
            maps_norm = (maps - maps.mean(axis=1, keepdims=True)) / (maps.std(axis=1, keepdims=True) + 1e-10)
        else:  # Cosine Similarity
            maps_norm = maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-10)

        batch_count = int(np.ceil(n_samples / batch_size))
        for b, batch_start in enumerate(range(0, n_samples, batch_size)):
            # Check for stop during final processing
            if worker and hasattr(worker, 'stopped') and worker.stopped:
                if verbose:
                    print(f"\n[CLUSTERING] TAAHC stopped during final processing")
                return None, np.inf

            if verbose and (b % 20 == 0 or b == batch_count - 1):
                print(f"[CLUSTERING] Final processing batch {b + 1}/{batch_count}")

            # Update progress for final batches
            if b % max(1, batch_count // 10) == 0:  # Update every 10%
                progress_pct = int((b / batch_count) * 100)
                update_progress(f"Final residual calculation... {progress_pct}% complete")

            batch_end = min(batch_start + batch_size, n_samples)
            batch_data = data[:, batch_start:batch_end]

            # Normalize based on the chosen metric
            if metric == 'Spatial Correlation':
                batch_norm = (batch_data - batch_data.mean(axis=0)) / (batch_data.std(axis=0, ddof=1) + 1e-10)
            else:  # Cosine Similarity
                batch_norm = batch_data / (np.linalg.norm(batch_data, axis=0, keepdims=True) + 1e-10)

            batch_corrs = np.abs(np.dot(maps_norm, batch_norm))
            final_corrs[batch_start:batch_end] = np.max(batch_corrs, axis=0)

        residual = 1 - np.mean(final_corrs)

        # Final statistics for each cluster
        if verbose:
            update_progress("Computing final cluster statistics...")

            final_assignments = np.zeros(n_samples, dtype=int)
            for batch_start in range(0, n_samples, batch_size):
                # Check for stop during final statistics
                if worker and hasattr(worker, 'stopped') and worker.stopped:
                    if verbose:
                        print(f"\n[CLUSTERING] TAAHC stopped during final statistics")
                    # Return None for maps since we don't have the correct number of states
                    return None, np.inf
                    
                batch_end = min(batch_start + batch_size, n_samples)
                batch_data = data[:, batch_start:batch_end]

                # Final batch normalization for assignments
                if metric == 'Spatial Correlation':
                    batch_norm = (batch_data - batch_data.mean(axis=0)) / (batch_data.std(axis=0, ddof=1) + 1e-10)
                else:  # Cosine Similarity
                    batch_norm = batch_data / (np.linalg.norm(batch_data, axis=0, keepdims=True) + 1e-10)

                batch_corrs = np.abs(np.dot(maps_norm, batch_norm))
                final_assignments[batch_start:batch_end] = np.argmax(batch_corrs, axis=0)

            total_time = time.time() - start_time
            print("=" * 70)
            print(f"[CLUSTERING] TAAHC clustering completed successfully!")
            print(f"[CLUSTERING] Total time: {total_time:.2f} seconds")
            print(f"[CLUSTERING] Using similarity metric: {metric}")
            print(f"[CLUSTERING] Residual: {residual:.6f}")
            print(f"[CLUSTERING] Average iteration time: {np.mean(iteration_times):.3f}s")
            print("=" * 70)

        # Final progress update
        update_progress("TAAHC clustering completed successfully!")

        return maps, residual

    def _calculate_taahc_residual(self, data, maps, metric, batch_size, verbose=False, worker=None):
        """Helper method to calculate residual for TAAHC clustering."""
        n_samples = data.shape[1]
        
        # Final normalization based on the chosen metric
        if metric == 'Spatial Correlation':
            maps_norm = (maps - maps.mean(axis=1, keepdims=True)) / (maps.std(axis=1, keepdims=True) + 1e-10)
        else:  # Cosine Similarity
            maps_norm = maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-10)

        # Calculate final residual - using batches
        final_corrs = np.zeros(n_samples)
        batch_count = int(np.ceil(n_samples / batch_size))
        
        for b, batch_start in enumerate(range(0, n_samples, batch_size)):
            # Check for stop during residual calculation
            if worker and hasattr(worker, 'stopped') and worker.stopped:
                if verbose:
                    print(f"[CLUSTERING] TAAHC stopped during residual calculation")
                return 1.0  # Return worst case residual
            
            if verbose and (b % 20 == 0 or b == batch_count - 1):
                print(f"[CLUSTERING] Residual calculation batch {b + 1}/{batch_count}")

            batch_end = min(batch_start + batch_size, n_samples)
            batch_data = data[:, batch_start:batch_end]

            # Normalize based on the chosen metric
            if metric == 'Spatial Correlation':
                batch_norm = (batch_data - batch_data.mean(axis=0)) / (batch_data.std(axis=0, ddof=1) + 1e-10)
            else:  # Cosine Similarity
                batch_norm = batch_data / (np.linalg.norm(batch_data, axis=0, keepdims=True) + 1e-10)

            batch_corrs = np.abs(np.dot(maps_norm, batch_norm))
            final_corrs[batch_start:batch_end] = np.max(batch_corrs, axis=0)

        return 1 - np.mean(final_corrs)

    # --------------------------------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------------------------------

    @staticmethod
    def corr_vectors(array1, array2, axis=0):
        """
        Compute the Pearson correlation between two matrices along a specified axis.

        Args:
            array1 (ndarray): First matrix.
            array2 (ndarray): Second matrix.
            axis (int): Axis along which to compute the correlation.

        Returns:
            ndarray: Pearson correlation coefficients.

        Notes:
            This implementation centers and normalizes the arrays before computing
            the correlation, making it equivalent to the Pearson correlation coefficient.
        """
        # Center and normalize matrices
        array1n = array1 - np.mean(array1, axis=axis, keepdims=True)
        array1n /= np.linalg.norm(array1n, axis=axis, ord=2, keepdims=True)
        array2n = array2 - np.mean(array2, axis=axis, keepdims=True)
        array2n /= np.linalg.norm(array2n, axis=axis, ord=2, keepdims=True)

        return np.sum(array1n * array2n, axis=axis)

    def compute_gev(self, data, maps):
        """
        Calculate the global explained variance (GEV) of microstate maps.

        GEV measures how well a set of microstate maps explains the variance
        in the EEG data, with values ranging from 0 to 1 (higher is better).

        Args:
            data (ndarray): Input EEG data with shape (n_channels, n_samples) or
                (n_samples, n_channels).
            maps (ndarray): Microstate maps with shape (n_states, n_channels).

        Returns:
            float: Global explained variance, between 0 and 1.

        Notes:
            This implementation automatically handles different input orientations.
            The formula is: GEV = sum((GFP * correlation_to_best_map)²) / sum(GFP²)
        """
        # Handle data orientation
        if len(maps.shape) > 1 and data.shape[0] != maps.shape[1]:
            data = data.T

        # Calculate GFP (Global Field Power)
        gfp = np.std(data, axis=0)

        # Normalize maps
        if maps.ndim == 1:
            maps = maps / np.linalg.norm(maps)
            maps = np.reshape(maps, (1, -1))
        else:
            maps = maps / np.linalg.norm(maps, axis=1, keepdims=True)

        # Calculate activation and segmentation
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Get maps for each time point
        selected_maps = maps[segmentation, :]

        # Calculate correlation between data and selected maps
        map_corr = self.corr_vectors(data, selected_maps.T)

        # Calculate GEV
        return np.sum((gfp * map_corr) ** 2) / np.sum(gfp ** 2)


# wrapper removed

