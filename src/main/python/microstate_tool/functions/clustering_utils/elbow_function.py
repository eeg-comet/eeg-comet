"""
Last Modified: April 18th, 2023
Description: This file provides functions for performing Modified K-Means clustering_utils on EEG data.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""
import numpy as np
import random
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import silhouette_score
from functions.clustering_utils.microstate_clusterer import run_modified_kmeans
from functions.utils.corr_vectors import corr_vectors
from sklearn.mixture import GaussianMixture

def get_elbow(data, min_dist, tolerance, n_inits, kmin, kmax, ax1, ax2, ax3):
    """
    Perform Modified K-Means clustering_utils on EEG data and compute the elbow point using the Silhouette score.

    Inputs:
        data (ndarray): EEG data of shape (n_channels, n_samples).
        min_dist (float): The minimum distance between the clusters.
        tolerance (float): The convergence threshold for the algorithm.
        n_inits (int): The number of initializations for the Modified K-Means algorithm.
        kmin (int): The minimum number of clusters to test.
        kmax (int): The maximum number of clusters to test.
        ax1 (matplotlib axis): The axis to plot the residual plot.
        ax2 (matplotlib axis): The axis to plot the global explained variance plot.
        ax3 (matplotlib axis): The axis to plot the Silhouette score plot.

    Outputs:
        None
    """
    '''
    # Randomly choose 10 percent of data for computation of Silhouette score
    subset_data = data
    # generate a random starting index
    rnd_len = int(0.1 * data.shape[1])
    start_index = random.randint(0, data.shape[1] - rnd_len)
    # select a continuous subset of 10 values
    subset_data = subset_data[:, start_index:start_index+rnd_len]
    '''
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp)
    all_maps = data.T
    all_maps /= np.linalg.norm(all_maps, axis=1, keepdims=True)

    SIL = []
    N, RES, GEV = [], [], []
    for k in range(kmin, kmax+1):
        print('\nClustering data with', k, 'microstates')
        gev_i, residual_i = 0, 0
        for init in range(n_inits):
            maps, gev, residual = run_modified_kmeans(data=data,
                                                     min_dist=min_dist,
                                                     n_states=k,
                                                     thresh=tolerance,
                                                     n_inits=1,
                                                     initializer="Random")
            gev_i = gev_i + gev
            residual_i = residual_i + residual
            # Compute the Silhouette score
            activation = np.array(maps).dot(data)
            classes = np.argmax(np.abs(activation), axis=0)
            sil_score_i = np.mean(abs(corr_vectors(data, maps[classes].T)))

        N = np.append(N, k)
        residual_mean = residual_i / n_inits
        RES = np.append(RES, residual_mean)
        gev_mean = gev_i / n_inits
        GEV = np.append(GEV, gev_mean)
        sil_mean = sil_score_i / n_inits
        SIL = np.append(SIL, sil_mean)
        '''
        # AIC BIC
        # Create empty dictionary for AIC and BIC values
        aic_score = {}
        bic_score = {}
        # Create Gaussian Mixture Model
        gmm = GaussianMixture(n_components=k,
                              tol=tolerance,
                              random_state=0,
                              init_params='random_from_data').fit(maps)
        # Get AIC score for the model
        aic_score[k] = -gmm.aic(maps)
        # Get BIC score for the model
        bic_score[k] = -gmm.bic(maps)

        print(list(aic_score.keys()))
        print(list(aic_score.values()))
        print(list(bic_score.keys()))
        print(list(bic_score.values()))
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.plot(list(aic_score.keys()), list(aic_score.values()), label='AIC')
    plt.plot(list(bic_score.keys()), list(bic_score.values()), label='BIC')
    plt.legend(loc='best')
    plt.title('AIC and BIC from GMM')
    plt.xlabel('Number of Clusters')
    plt.ylabel('AIC and BIC values')
    plt.show()
    '''

    print("Average GEVs for all K", GEV)
    data = np.column_stack((N, RES, GEV, SIL))
    df = pd.DataFrame(data, columns=['K', 'Residual', 'Global Explained Variance', 'Silhouette Score'])

    # Visualize the result
    res_fig = sns.lineplot(data=df, x="K", y="Residual",
                           linewidth=5, marker="o", markersize=16, dashes=False, ax=ax1)
    res_fig.set_xlabel("Number of Microstate Maps", size=16)
    res_fig.set_ylabel("Residual", size=16)

    xlabel_format = '{:,.0f}'
    ticks_loc = ax1.get_xticks().tolist()
    ax1.set_xticks(ax1.get_xticks().tolist())
    ax1.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
    ylabel_format = '{:,.2f}'
    ticks_loc = ax1.get_yticks().tolist()
    ax1.set_yticks(ax1.get_yticks().tolist())
    ax1.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)


    gev_fig = sns.lineplot(data=df, x="K", y="Global Explained Variance",
                           linewidth=5, style=None, marker="o", markersize=16, dashes=False, ax=ax2)
    gev_fig.set_xlabel("Number of Microstate Maps", size=16)
    gev_fig.set_ylabel("Global Explained Variance", size=16)

    xlabel_format = '{:,.0f}'
    ticks_loc = ax2.get_xticks().tolist()
    ax2.set_xticks(ax2.get_xticks().tolist())
    ax2.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
    ylabel_format = '{:,.2f}'
    ticks_loc = ax2.get_yticks().tolist()
    ax2.set_yticks(ax2.get_yticks().tolist())
    ax2.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)

    silhouette_fig = sns.lineplot(data=df, x="K", y="Silhouette Score",
                           linewidth=5, style=None, marker="o", markersize=16, dashes=False, ax=ax3)
    silhouette_fig.set_xlabel("Number of Microstate Maps", size=16)
    silhouette_fig.set_ylabel("Silhouette Score", size=16)

    xlabel_format = '{:,.0f}'
    ticks_loc = ax3.get_xticks().tolist()
    ax3.set_xticks(ax3.get_xticks().tolist())
    ax3.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], size=14)
    ylabel_format = '{:,.2f}'
    ticks_loc = ax3.get_yticks().tolist()
    ax3.set_yticks(ax3.get_yticks().tolist())
    ax3.set_yticklabels([ylabel_format.format(x) for x in ticks_loc], size=14)



def get_elbow_without_plt(data, min_dist, tolerance, n_inits, kmin, kmax, stopping_mode='gev', threshold=0.1):
    # change thresholf to percentage
    if threshold > 1:
        threshold /= 100
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp)
    all_maps = data.T
    all_maps /= np.linalg.norm(all_maps, axis=1, keepdims=True)

    SIL = []
    N, RES, GEV = [], [], []
    for k in range(kmin, kmax+1):
        print('\nClustering data with', k, 'microstates')
        gev_i, residual_i = 0, 0
        for init in range(n_inits):
            maps, gev, residual = run_modified_kmeans(data=data,
                                                     min_dist=min_dist,
                                                     n_states=k,
                                                     thresh=tolerance,
                                                     n_inits=1,
                                                     initializer="Random")
            gev_i = gev_i + gev
            residual_i = residual_i + residual
            # Compute the Silhouette score
            activation = np.array(maps).dot(data)
            classes = np.argmax(np.abs(activation), axis=0)
            sil_score_i = np.mean(abs(corr_vectors(data, maps[classes].T)))

        N = np.append(N, k)
        residual_mean = residual_i / n_inits
        RES = np.append(RES, residual_mean)
        gev_mean = gev_i / n_inits
        GEV = np.append(GEV, gev_mean)
        sil_mean = sil_score_i / n_inits
        SIL = np.append(SIL, sil_mean)
        if len(N) == 1:
            continue
        if stopping_mode=='gev':
            if abs(gev_mean - GEV[-2]) / GEV[-2] < threshold:
                return k
        elif stopping_mode=='residual':
            if abs(RES[-2] - residual_mean) / RES[-2]  < threshold:
                return k
        elif stopping_mode=='sil':
            if abs(sil_mean - SIL[-2]) / SIL[-2] < threshold:
                return k


    return int((kmin + kmax)/2)





