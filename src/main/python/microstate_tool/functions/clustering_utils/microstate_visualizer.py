
import mne

def show_microstate(microstate_map, eeg_info, ax, polarity=1, sensors=False, contours=6, cmap=None):
    """
        Display a topographic map of microstate data using MNE-Python.

        Parameters:
        - microstate_map (numpy array): Microstate data to be visualized.
        - eeg_info (instance of Info): EEG data information, containing channel locations and types.
        - ax (instance of Axes): Matplotlib axes where the topomap will be plotted.
        - polarity (int, optional): Multiplier for the microstate map values to control color polarity. Default is 1.
        - sensors (bool or str, optional): Whether to add markers for sensor locations. If True, black circles will be used.
          If str, a valid matplotlib format string can be provided. Default is False.
        - contours (int or array_like, optional): The number of contour lines to draw on the topomap. Default is 6.
        - cmap (str or None, optional): Colormap to use for the topomap. Options include 'RdBu_r', 'coolwarm', 'bwr', 'seismic', among others.
          Default is None.
    """
    mne.viz.plot_topomap(polarity*microstate_map, eeg_info, sensors=sensors, contours=contours,
                         axes=ax, cmap=cmap, show=False)
