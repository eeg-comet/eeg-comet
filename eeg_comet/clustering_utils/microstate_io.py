"""Import/export utilities for microstate maps (EEG-COMET)."""

import numpy as np
import pandas as pd


class MicrostateIO:
    """Import/export utilities for microstate maps and channels."""

    def __init__(self):
        """Initialize a new MicrostateIO instance."""
        return

    @staticmethod
    def export_microstates(microstate_maps, eeg_info, microstate_maps_path, headers=None):
        """Export microstate maps to a CSV file.

        Args:
            microstate_maps (ndarray): Microstate maps with shape (n_states, n_channels)
                or (n_channels, n_states).
            eeg_info (dict): Dictionary containing EEG channel information with at least
                a 'ch_names' key for channel names.
            microstate_maps_path (str): Path where the CSV file will be saved.
            headers (list, optional): Column headers for the CSV. If None, columns
                will be numbered as '1', '2', etc.

        Notes:
            The function automatically detects map orientation and transposes if needed
            to ensure channels are on rows and microstate maps are on columns.
            The first column in the CSV will be labeled 'channel' and contain channel names.
        """
        if not isinstance(microstate_maps, np.ndarray):
            microstate_maps = np.array(microstate_maps)

        # Transpose the microstates array if needed
        # (ensure channels are on rows and microstates are on columns)
        if microstate_maps.shape[1] == len(eeg_info["ch_names"]):
            microstate_maps = microstate_maps.T

        # Create DataFrame with channel names as index
        maps_df = pd.DataFrame(microstate_maps, index=eeg_info["ch_names"])

        # Generate column headers if not provided
        if headers is None:
            headers = [f"{i + 1}" for i in range(microstate_maps.shape[1])]
        maps_df.columns = headers

        # Save to CSV with 'channel' as the first column header
        maps_df.index.name = "channel"
        maps_df.to_csv(microstate_maps_path)

    @staticmethod
    def load_microstates(microstate_maps_path):
        """Load microstate maps from a CSV file.

        Args:
            microstate_maps_path (str): Path to the CSV file containing microstate maps.

        Returns:
            tuple or ndarray: If return_microstate_labels is True, returns a tuple containing:
                - microstate_maps (ndarray): Microstate maps with shape (n_channels, n_states)
                - microstate_labels (list): List of microstate labels (column headers)
                If return_microstate_labels is False, returns a tuple with microstate_maps and ch_names.

        Notes:
            The function expects CSV format where rows represent channels and columns
            represent microstate maps. The index column is assumed to contain channel names
            with a header of 'channel'.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file cannot be parsed as a valid microstate map CSV.
        """
        try:
            # Read CSV file into DataFrame
            maps_df = pd.read_csv(microstate_maps_path, index_col=0)

            # Convert DataFrame to numpy array
            microstate_maps = maps_df.values

            # Get microstate labels from column headers
            microstate_labels = maps_df.columns.tolist()
            return microstate_maps.T, microstate_labels

        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"Microstate maps file not found: {microstate_maps_path}"
            ) from err
        except Exception as err:
            raise ValueError(f"Error parsing microstate maps CSV: {str(err)}") from err
