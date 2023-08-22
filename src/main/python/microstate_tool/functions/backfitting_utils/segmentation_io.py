
"""
This script defines a class called SegmentationIO that provides functions to export and import segmentation data
in various formats ('csv', 'pkl', 'hdf', or 'json'). The exported data consists of a filename and a numpy array
called str_segmentation, containing an array of strings.

"""

import os
import numpy as np
import pandas as pd
import pickle
import h5py
import json

class SegmentationIO:
    def __init__(self):
        pass

    @staticmethod
    def export_segmentation(output_folder, filename, segmentation_array, time_array, export_format='csv'):
        """
        Export segmentation data along with time array to a file in the specified format.

        Args:
            output_folder (str): Folder path to export the file.
            filename (str): Name of the file to export (without extension).
            segmentation_array (np.ndarray): Numpy array of strings containing segmentation data.
            time_array (np.ndarray): Numpy array of time values.
            export_format (str): Format for exporting ('csv', 'pkl', 'hdf', or 'json').

        Returns:
            bool: True if export is successful, False otherwise.
        """
        valid_formats = ['csv', 'pkl', 'hdf', 'json']
        if export_format not in valid_formats:
            print("Invalid export format.")
            return False

        output_path = os.path.join(output_folder, f"{filename}.{export_format}")

        try:
            if export_format == 'csv':
                df = pd.DataFrame({'time': time_array, 'segmentation': segmentation_array})
                df.to_csv(output_path, index=False)
            elif export_format == 'pkl':
                with open(output_path, 'wb') as f:
                    pickle.dump((time_array, segmentation_array), f)
            elif export_format == 'hdf':
                with h5py.File(output_path, 'w') as f:
                    f.create_dataset('time', data=time_array)
                    f.create_dataset('segmentation', data=segmentation_array)
            elif export_format == 'json':
                with open(output_path, 'w') as f:
                    json.dump({'time': time_array.tolist(), 'segmentation': segmentation_array.tolist()}, f)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

    @staticmethod
    def load_segmentation(filename, import_format='csv'):
        """
        Load segmentation data from a file.

        Args:
            filename (str): Name of the file to load (including extension).
            import_format (str): Format of the imported file ('csv', 'pkl', 'hdf', or 'json').

        Returns:
            np.ndarray or None: Loaded segmentation data or None if loading fails.
        """
        valid_formats = ['csv', 'pkl', 'hdf', 'json']
        if import_format not in valid_formats:
            print("Invalid import format.")
            return None

        try:
            with open(filename, 'rb') as f:
                if import_format == 'csv':
                    df = pd.read_csv(filename)
                    return np.array(df['segmentation'])
                elif import_format == 'pkl':
                    return pickle.load(f)
                elif import_format == 'hdf':
                    with h5py.File(f, 'r') as h5f:
                        return np.array(h5f['segmentation'])
                elif import_format == 'json':
                    return np.array(json.load(f))
        except Exception as e:
            print(f"Import error: {e}")
            return None

