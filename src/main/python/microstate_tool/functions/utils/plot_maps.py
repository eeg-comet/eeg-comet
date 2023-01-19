
import mne
from matplotlib import pyplot as plt


def plot_map(microstate_map, eeg_info, polarity=1, cmap=None, sensors=False, show_names=False, contours=6):
    mne.viz.plot_topomap(polarity*microstate_map,
                         eeg_info,
                         cmap=cmap,
                         sensors=sensors,
                         show_names=show_names,
                         contours=contours)


def plot_maps(microstate_maps, eeg_info):
    #maps = np.array(maps)
    plt.figure(figsize=(2 * len(microstate_maps), 2))
    for i, microstate_map in enumerate(microstate_maps):
        plt.subplot(1, len(microstate_maps), i + 1)
        mne.viz.plot_topomap(microstate_map, eeg_info, sensors=False, show_names=False, contours=6)
        plt.title('%d' % i)

