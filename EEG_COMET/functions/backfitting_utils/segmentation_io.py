
import os
import numpy as np
import pandas as pd
import pickle
import h5py
import json


class SegmentationIO:
    """
    The SegmentationIO class provides methods for input/output operations related to segmentation.
    """
    def __init__(self):
        pass

    @staticmethod
    def export_segmentation(output_folder, filename, segmentation_array, time_array, export_format='.csv'):
        """
        Export segmentation data along with time array to a file in the specified format.

        Args:
            output_folder (str): The folder where the exported file will be saved.
            filename (str): The name of the exported file (without extension).
            segmentation_array (ndarray): The segmentation data to be exported.
            time_array (ndarray): The time array associated with the segmentation data.
            export_format (str, optional): The format of the exported file. Defaults to '.csv'.

        Returns:
            bool: True if the export is successful, False otherwise.
        """
        valid_formats = ['.csv', '.pkl', '.hdf', '.json']
        if export_format not in valid_formats:
            raise ValueError("Invalid export format.")

        output_path = os.path.join(output_folder, f"{filename}{export_format}")

        try:
            if export_format == '.csv':
                df = pd.DataFrame({'time': time_array, 'segmentation': segmentation_array})
                df.to_csv(output_path, index=False)
            elif export_format == '.pkl':
                with open(output_path, 'wb') as f:
                    pickle.dump((time_array, segmentation_array), f)
            elif export_format == '.hdf':
                with h5py.File(output_path, 'w') as f:
                    f.create_dataset('time', data=time_array)
                    f.create_dataset('segmentation', data=segmentation_array)
            elif export_format == '.json':
                with open(output_path, 'w') as f:
                    json.dump({'time': time_array.tolist(), 'segmentation': segmentation_array.tolist()}, f)
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

    @staticmethod
    def load_segmentation(segmentation_path, import_format='.csv'):
        """
        Load segmentation data from a file.

        Args:
            segmentation_path (str): The path to the segmentation file.
            import_format (str, optional): The format of the segmentation file. Defaults to '.csv'.

        Returns:
            ndarray or None: The loaded segmentation data if successful, None otherwise.

        Raises:
            ValueError: If the import format is not supported.
        """
        valid_formats = ['.csv', '.pkl', '.hdf', '.json']
        if import_format not in valid_formats:
            raise ValueError("Invalid import format.")

        try:
            with open(segmentation_path, 'rb') as f:
                if import_format == '.csv':
                    df = pd.read_csv(segmentation_path)
                    return np.array(df['segmentation'])
                elif import_format == '.pkl':
                    return pickle.load(f)
                elif import_format == '.hdf':
                    with h5py.File(f, 'r') as h5f:
                        return np.array(h5f['segmentation'])
                elif import_format == '.json':
                    return np.array(json.load(f))
        except Exception as e:
            print(f"Import error: {e}")
            return None
