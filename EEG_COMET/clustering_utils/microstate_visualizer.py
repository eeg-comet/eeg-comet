"""Visualization helpers for microstate topographies (EEG-COMET)."""

import mne
from mne.utils import use_log_level

# Global flag to track if electrode position warning has been shown
_electrode_warning_shown = False


def reset_electrode_warning():
    """Reset the electrode warning flag to allow showing the warning again.
    
    This should be called when loading a new study to ensure users get 
    one warning per study rather than per application session.
    """
    global _electrode_warning_shown
    _electrode_warning_shown = False


def show_microstate(microstate_map, eeg_info, ax, polarity=1, sensors=False, contours=6, cmap=None, log_callback=None):
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
        log_callback (callable, optional): Function to call for logging warnings.

    Returns:
        matplotlib.image.AxesImage: Handle to the image created by ``plot_topomap``.
    """
    try:
        # Suppress MNE warnings about electrode locations during visualization
        with use_log_level('ERROR'):
            im, _ = mne.viz.plot_topomap(
                polarity * microstate_map,
                eeg_info,
                sensors=sensors,
                contours=contours,
                axes=ax,
                cmap=cmap,
                show=False,
            )
    except RuntimeError as e:
        if "No digitization points found" in str(e):
            # Log a single warning about electrode position issues (only once per session)
            global _electrode_warning_shown
            if log_callback and not _electrode_warning_shown:
                log_callback("Electrode positions not found. Applying standard montage for microstate visualization.")
                _electrode_warning_shown = True
            
            # Fallback: try to apply a standard montage based on available channels
            try:
                # Create a copy of the info to avoid modifying the original
                info_copy = eeg_info.copy()
                
                # Try to apply a standard montage that matches the channels
                standard_montage = mne.channels.make_standard_montage('standard_1020')
                
                # Filter the montage to only include channels present in the data
                available_channels = [ch for ch in info_copy.ch_names if ch in standard_montage.ch_names]
                if available_channels:
                    # Pick only the available channels from both info and montage
                    picks = mne.pick_channels(info_copy.ch_names, available_channels)
                    info_subset = mne.pick_info(info_copy, picks)
                    
                    # Apply montage to the subset
                    info_subset.set_montage(standard_montage, match_case=False, on_missing='ignore')
                    
                    # Subset the microstate map as well
                    microstate_subset = microstate_map[picks]
                    
                    # Suppress warnings in fallback visualization too
                    with use_log_level('ERROR'):
                        im, _ = mne.viz.plot_topomap(
                            polarity * microstate_subset,
                            info_subset,
                            sensors=sensors,
                            contours=contours,
                            axes=ax,
                            cmap=cmap,
                            show=False,
                        )
                else:
                    # If no standard channels found, display an error message on the plot
                    ax.text(0.5, 0.5, 'No electrode\npositions\navailable', 
                           ha='center', va='center', fontsize=12, transform=ax.transAxes)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    im = ax.imshow([[0]], alpha=0)  # Invisible dummy image
            except Exception:
                # Final fallback: display an error message
                ax.text(0.5, 0.5, 'Visualization\nerror', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                im = ax.imshow([[0]], alpha=0)  # Invisible dummy image
        else:
            raise e

    return im
