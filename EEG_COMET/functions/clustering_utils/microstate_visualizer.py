
import mne


def show_microstate(microstate_map, eeg_info, ax, polarity=1, sensors=False, contours=6, cmap=None):
    """
        Display a topographic map of microstate data using MNE-Python.
    """
    mne.viz.plot_topomap(
        polarity*microstate_map, eeg_info, sensors=sensors, contours=contours, axes=ax, cmap=cmap, show=False)
