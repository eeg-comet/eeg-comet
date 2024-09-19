
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
            if segmentation_array.shape[0] > 1:
                data = {'trial': [], 'time': [], 'label': []}
                for trial in range(segmentation_array.shape[0]):
                    for time_idx, time_val in enumerate(time_array):
                        data['trial'].append(trial + 1)
                        data['time'].append(time_val)
                        data['label'].append(segmentation_array[trial, time_idx])
                if export_format == '.csv':
                    df = pd.DataFrame(data)
                    df.to_csv(output_path, index=False)
                elif export_format == '.pkl':
                    with open(output_path, 'wb') as f:
                        pickle.dump(data, f)
                elif export_format == '.hdf':
                    with h5py.File(output_path, 'w') as f:
                        f.create_dataset('trial', data=data['trial'])
                        f.create_dataset('time', data=data['time'])
                        f.create_dataset('label', data=data['label'])
                elif export_format == '.json':
                    with open(output_path, 'w') as f:
                        json.dump(data, f)
            else:
                if export_format == '.csv':
                    df = pd.DataFrame({'time': time_array, 'label': segmentation_array[0]})
                    df.to_csv(output_path, index=False)
                elif export_format == '.pkl':
                    with open(output_path, 'wb') as f:
                        pickle.dump({'time': time_array, 'label': segmentation_array[0]}, f)
                elif export_format == '.hdf':
                    with h5py.File(output_path, 'w') as f:
                        f.create_dataset('time', data=time_array)
                        f.create_dataset('label', data=segmentation_array[0])
                elif export_format == '.json':
                    with open(output_path, 'w') as f:
                        json.dump({'time': time_array.tolist(), 'label': segmentation_array[0].tolist()}, f)
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
            ndarray: The loaded segmentation array with shape (number of trials, number of time points).
        """
        valid_formats = ['.csv', '.pkl', '.hdf', '.json']
        if import_format not in valid_formats:
            raise ValueError("Invalid import format.")

        try:
            if import_format == '.csv':
                df = pd.read_csv(segmentation_path)
                if 'trial' in df.columns:
                    # Get unique trials and time points
                    trials = df['trial'].unique()
                    time_points = df['time'].unique()

                    # Create an empty array to store segmentation data (as object type to handle strings)
                    segmentation_array = np.empty((len(trials), len(time_points)), dtype=object)

                    for trial in trials:
                        trial_data = df[df['trial'] == trial]
                        segmentation_array[trial - 1, :] = trial_data['label'].values

                    return segmentation_array

                else:
                    # In case there's only one trial (no 'trial' column)
                    return np.array([df['label'].values], dtype=object)

            elif import_format == '.pkl':
                with open(segmentation_path, 'rb') as f:
                    data = pickle.load(f)
                    if 'trial' in data:
                        trials = np.unique(data['trial'])
                        time_points = np.unique(data['time'])

                        segmentation_array = np.empty((len(trials), len(time_points)), dtype=object)
                        for i, trial in enumerate(trials):
                            indices = np.where(data['trial'] == trial)
                            segmentation_array[i, :] = np.array(data['label'])[indices]

                        return segmentation_array
                    else:
                        return np.array([data['label']], dtype=object)

            elif import_format == '.hdf':
                with h5py.File(segmentation_path, 'r') as h5f:
                    if 'trial' in h5f.keys():
                        trials = np.unique(h5f['trial'])
                        time_points = np.unique(h5f['time'])

                        segmentation_array = np.empty((len(trials), len(time_points)), dtype=object)
                        for i, trial in enumerate(trials):
                            indices = np.where(h5f['trial'][:] == trial)
                            segmentation_array[i, :] = h5f['label'][indices].astype(str)

                        return segmentation_array
                    else:
                        return np.array([h5f['label'][:].astype(str)], dtype=object)

            elif import_format == '.json':
                with open(segmentation_path, 'r') as f:
                    data = json.load(f)
                    if 'trial' in data:
                        trials = np.unique(data['trial'])
                        time_points = np.unique(data['time'])

                        segmentation_array = np.empty((len(trials), len(time_points)), dtype=object)
                        for i, trial in enumerate(trials):
                            indices = np.where(np.array(data['trial']) == trial)
                            segmentation_array[i, :] = np.array(data['label'])[indices].astype(str)

                        return segmentation_array
                    else:
                        return np.array([data['label']], dtype=object)

        except Exception as e:
            print(f"Import error: {e}")
            return None


