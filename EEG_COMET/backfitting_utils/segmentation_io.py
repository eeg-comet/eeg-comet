"""Segmentation import/export helpers for EEG-COMET."""

import json
import os

import h5py
import numpy as np
import pandas as pd

from data_utils.safe_io import safe_pickle_load


class SegmentationIO:
    """I/O operations for segmentation arrays and metadata."""

    def __init__(self):
        """Initialize a new SegmentationIO instance."""
        return

    @staticmethod
    def export_segmentation(
        output_folder, filename, segmentation_array, time_array, export_format=".csv"
    ):
        """Export segmentation data along with time array to a file in the specified format.

        Args:
            output_folder (str): The folder where the exported file will be saved.
            filename (str): The name of the exported file (without extension).
            segmentation_array (ndarray): The segmentation data to be exported.
            time_array (ndarray): The time array associated with the segmentation data.
            export_format (str, optional): The format of the exported file. Defaults to '.csv'.

        Returns:
            bool: True if the export is successful, False otherwise.
        """
        valid_formats = [".csv", ".pkl", ".hdf", ".json"]
        if export_format not in valid_formats:
            raise ValueError("Invalid export format.")

        output_path = os.path.join(output_folder, f"{filename}{export_format}")

        try:
            if len(segmentation_array.shape) == 2:
                data = {"trial": [], "time": [], "label": []}
                for trial in range(segmentation_array.shape[0]):
                    for time_idx, time_val in enumerate(time_array):
                        data["trial"].append(trial + 1)
                        data["time"].append(time_val)
                        data["label"].append(segmentation_array[trial, time_idx])
                if export_format == ".csv":
                    df = pd.DataFrame(data)
                    df.to_csv(output_path, index=False)
                elif export_format == ".pkl":
                    with open(output_path, "wb") as f:
                        pickle.dump(data, f)
                elif export_format == ".hdf":
                    with h5py.File(output_path, "w") as f:
                        f.create_dataset("trial", data=data["trial"])
                        f.create_dataset("time", data=data["time"])
                        f.create_dataset("label", data=data["label"])
                elif export_format == ".json":
                    with open(output_path, "w") as f:
                        json.dump(data, f)
            else:
                if export_format == ".csv":
                    df = pd.DataFrame({"time": time_array, "label": segmentation_array})
                    df.to_csv(output_path, index=False)
                elif export_format == ".pkl":
                    with open(output_path, "wb") as f:
                        pickle.dump({"time": time_array, "label": segmentation_array}, f)
                elif export_format == ".hdf":
                    with h5py.File(output_path, "w") as f:
                        f.create_dataset("time", data=time_array)
                        f.create_dataset("label", data=segmentation_array)
                elif export_format == ".json":
                    with open(output_path, "w") as f:
                        json.dump(
                            {"time": time_array.tolist(), "label": segmentation_array.tolist()}, f
                        )
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

    @staticmethod
    def load_segmentation(segmentation_path, import_format=".csv"):
        """Load segmentation data from a file.

        Args:
            segmentation_path (str): The path to the segmentation file.
            import_format (str, optional): The format of the segmentation file. Defaults to '.csv'.

        Returns:
            ndarray: The loaded segmentation array with shape (number of trials, number of time points).
        """
        valid_formats = [".csv", ".pkl", ".hdf", ".json"]
        if import_format not in valid_formats:
            raise ValueError("Invalid import format.")

        try:
            if import_format == ".csv":
                df = pd.read_csv(segmentation_path)
                if "trial" in df.columns:
                    trials = df["trial"].unique()
                    time_points = df["time"].unique()
                    # Use string dtype to ensure hashable labels
                    segmentation_array = np.empty((len(trials), len(time_points)), dtype="<U10")
                    for trial in trials:
                        trial_data = df[df["trial"] == trial]
                        # Convert labels to strings and ensure proper ordering by time
                        trial_data_sorted = trial_data.sort_values("time")
                        labels = trial_data_sorted["label"].astype(str).values
                        segmentation_array[trial - 1, :] = labels
                    return segmentation_array
                # For non-epoched data, also convert to string
                labels = df["label"].astype(str).values
                return np.array([labels], dtype="<U10")
            if import_format == ".pkl":
                data = safe_pickle_load(segmentation_path)
                if "trial" in data:
                    trials = np.unique(data["trial"])
                    time_points = np.unique(data["time"])
                    # Use string dtype to ensure hashable labels
                    segmentation_array = np.empty((len(trials), len(time_points)), dtype="<U10")
                    for i, trial in enumerate(trials):
                        indices = np.where(data["trial"] == trial)
                        labels = np.array(data["label"])[indices].astype(str)
                        segmentation_array[i, :] = labels
                    return segmentation_array
                labels = np.array(data["label"]).astype(str)
                return np.array([labels], dtype="<U10")
            elif import_format == ".hdf":
                with h5py.File(segmentation_path, "r") as h5f:
                    if "trial" in h5f:
                        trials = np.unique(h5f["trial"])
                        time_points = np.unique(h5f["time"])
                        # Use string dtype to ensure hashable labels
                        segmentation_array = np.empty((len(trials), len(time_points)), dtype="<U10")
                        for i, trial in enumerate(trials):
                            indices = np.where(h5f["trial"][:] == trial)
                            labels = h5f["label"][indices].astype(str)
                            segmentation_array[i, :] = labels
                        return segmentation_array
                    labels = h5f["label"][:].astype(str)
                    return np.array([labels], dtype="<U10")
            elif import_format == ".json":
                with open(segmentation_path) as f:
                    data = json.load(f)
                    if "trial" in data:
                        trials = np.unique(data["trial"])
                        time_points = np.unique(data["time"])
                        # Use string dtype to ensure hashable labels
                        segmentation_array = np.empty((len(trials), len(time_points)), dtype="<U10")
                        for i, trial in enumerate(trials):
                            indices = np.where(np.array(data["trial"]) == trial)
                            labels = np.array(data["label"])[indices].astype(str)
                            segmentation_array[i, :] = labels
                        return segmentation_array
                    labels = np.array(data["label"]).astype(str)
                    return np.array([labels], dtype="<U10")

        except Exception as e:
            print(f"Import error: {e}")
            return None
