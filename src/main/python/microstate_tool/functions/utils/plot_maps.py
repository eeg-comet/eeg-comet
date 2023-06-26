"""
Last Modified: April 18th, 2023
Description: This file defines two functions for plotting microstate maps from EEG data using the MNE library.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import mne
from matplotlib import pyplot as plt


def plot_map(microstate_map, eeg_info, polarity=1, cmap=None, sensors=False, show_names=False, contours=6):
    """
    Plots a single microstate map using the MNE library.

    Inputs:
        microstate_map (numpy array): The microstate map to plot.
        eeg_info (mne.Info object): The MNE Info object containing information about the EEG data.
        polarity (int, optional): The polarity of the microstate map. Default is 1.
        cmap (str, optional): The colormap to use for plotting the microstate map. Default is None.
        sensors (bool, optional): Whether or not to plot the EEG sensors on the map. Default is False.
        show_names (bool, optional): Whether or not to show the channel names on the map. Default is False.
        contours (int, optional): The number of contour lines to use when plotting the map. Default is 6.

    Outputs:
        None
    """
    mne.viz.plot_topomap(polarity*microstate_map,
                         eeg_info,
                         cmap=cmap,
                         sensors=sensors,
                         show_names=show_names,
                         contours=contours)


def plot_maps(microstate_maps, eeg_info):
    """
    Plots multiple microstate maps using the MNE library.

    Inputs:
        microstate_maps (list of numpy arrays): The microstate maps to plot.
        eeg_info (mne.Info object): The MNE Info object containing information about the EEG data.

    Outputs:
        None
    """
    plt.figure(figsize=(2 * len(microstate_maps), 2))
    for i, microstate_map in enumerate(microstate_maps):
        plt.subplot(1, len(microstate_maps), i + 1)
        mne.viz.plot_topomap(microstate_map, eeg_info, sensors=False, show_names=False, contours=6)
        plt.title('%d' % i)
