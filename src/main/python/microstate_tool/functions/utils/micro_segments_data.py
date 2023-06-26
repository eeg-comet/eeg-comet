"""
Last Modified: April 18th, 2023
Description: This file defines a function for extracting micro-segments from EEG data using HDF5 files.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import os.path
import numpy as np
import h5py


def micro_segments_data(hf_data_path, hf_segmentation_path, n_chan, micro_labels, save_path):
    """
    Extracts micro-segments from EEG data and saves the results in HDF5 files.

    Inputs:
        hf_data_path (string): Path to the HDF5 file containing the EEG data.
        hf_segmentation_path (string): Path to the HDF5 file containing the EEG segmentation data.
        n_chan (int): Number of EEG channels.
        micro_labels (list of strings): List of microstate labels to extract.
        save_path (string): Path to the directory where the extracted micro-segments should be saved.

    Outputs:
        None
    """
    # Load data and segment files
    hf_data = h5py.File(hf_data_path, 'r')  # Open the HDF5 file containing the EEG data for reading
    hf_segmentation = h5py.File(hf_segmentation_path, 'r')  # Open the HDF5 file containing the EEG segmentation data for reading
    study_name = list(hf_data.keys())[0]  # Get the study name from the HDF5 file
    file_names = list(hf_data[study_name].keys())  # Get the list of file names from the HDF5 file
    for micro_map in micro_labels:
        print("Extracting segments matching microstate ", micro_map)
        data_map_concatenated = np.empty((n_chan,0))  # Initialize an empty array to store the extracted micro-segments
        for filename in file_names:
            print(study_name, filename)
            data = np.asarray(hf_data[study_name][filename][:])  # Get the EEG data for the current file
            segment = np.asarray(hf_segmentation[study_name][filename][:])  # Get the segmentation data for the current file
            segment = [" ".join(item) for item in segment.astype('U10')]  # Convert the segmentation data to a string array
            segment = np.asarray(segment)  # Convert the segmentation data to a numpy array
            ind_map = np.where(segment == micro_map)[0]  # Find the indices of the segmentation data that match the current microstate
            data_map = data[:, ind_map]  # Extract the EEG data corresponding to the current microstate
            data_map_concatenated = np.concatenate((data_map_concatenated, data_map), axis=1)  # Concatenate the extracted data to the output array
        # save
        output_path = os.path.join(save_path, study_name+"_MicroSegment_"+micro_map+".hdf")  # Set the output file path
        f = h5py.File(output_path, "w")  # Open the output file for writing
        f.create_dataset(micro_map, data=data_map_concatenated, compression="gzip", compression_opts=9)  # Save the extracted micro-segments to the output file
        f.close()  # Close the output file
