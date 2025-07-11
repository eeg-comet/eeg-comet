import mne


def show_microstate(microstate_map, eeg_info, ax, polarity=1, sensors=False, contours=6, cmap=None):
    """Display a topographic map of microstate data using MNE-Python and return the
    image handle so that callers can attach a color-bar if needed.

    Args:
        microstate_map (np.ndarray): The microstate values for each channel.
        eeg_info (mne.Info): Info structure describing sensors.
        ax (matplotlib.axes.Axes): Axis on which to plot.
        polarity (int, optional): Multiplier to flip polarity. Defaults to 1.
        sensors (bool, optional): Whether to show sensor positions. Defaults to False.
        contours (int, optional): Number of contour lines. Defaults to 6.
        cmap (str | matplotlib Colormap, optional): Colormap to use. Defaults to None.

    Returns:
        matplotlib.image.AxesImage: Handle to the image created by ``plot_topomap``.
    """
    im, _ = mne.viz.plot_topomap(
        polarity * microstate_map,
        eeg_info,
        sensors=sensors,
        contours=contours,
        axes=ax,
        cmap=cmap,
        show=False,
    )

    return im
