
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import silhouette_score
from functions import modified_kmeans


def get_elbow(data, kmin, kmax, ax1, ax2):

    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp)
    maps = data[:, peaks].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)

    N, RES, GEV = [], [], []
    for k in range(kmin, kmax+1):
        print(k)
        maps, gev, residual = modified_kmeans.segment(data=data,
                                                         peaks=peaks,
                                                         n_states=k,
                                                         n_inits=1,
                                                         max_n_peaks=None)

        N = np.append(N, k)
        RES = np.append(RES, residual)
        GEV = np.append(GEV, gev)

        #activation = np.array(maps).dot(data)
        #segmentation = np.argmax(np.abs(activation), axis=0)
        #data_df = pd.DataFrame(data.T)
        #SILHOUETTE_SCORE = silhouette_score(data_df, segmentation)
        #print(SILHOUETTE_SCORE)

    data = np.column_stack((N, RES, GEV))
    df = pd.DataFrame(data, columns=['K', 'Residual', 'Global Explained Variance'])

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
