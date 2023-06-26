"""
Last Modified: April 18th, 2023
Description: This file defines a function for saving a pandas dataframe to a file with a specified format and location.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import os.path


def save_features(df, filename, file_format, path):
    """
    Saves a pandas dataframe to a file with a specified format and location.

    Inputs:
        df (pandas dataframe): The dataframe to save.
        filename (string): The filename to use for the saved file.
        file_format (string): The file format to use for the saved file (e.g., '.csv', '.pkl', '.hdf', '.json').
        path (string): The directory path to save the file to.

    Outputs:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory if it doesn't exist
    save_path = os.path.join(path, filename + file_format)  # Construct the full save path
    if file_format == '.csv':
        df.to_csv(save_path, header=True, index=False)  # Save the dataframe as a CSV file
    elif file_format == '.pkl':
        df.to_pickle(save_path)  # Save the dataframe as a pickled file
    elif file_format == '.hdf':
        df.to_hdf(save_path, key='df', mode='w')  # Save the dataframe as an HDF file
    elif file_format == '.json':
        df.to_json(save_path)  # Save the dataframe as a JSON file
