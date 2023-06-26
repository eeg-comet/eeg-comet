"""
Authors: Amin Kabir, Raaj Chatterjee, Faranak Farzan
Organization: SFU eBrain Lab, www.ebrainlab.ca
Last Modified: April 18th, 2023
"""

import os.path
import h5py


def export_h5(data, filename, ch_names, eeg_format, data_type, filter_method,
              lowcut_freq, highcut_freq, sample_rate, ch2rm, path):
    """
    Exports EEG data to an HDF5 file.

    Inputs:
        data (ndarray): The EEG data.
        filename (string): The name of the file to be exported.
        ch_names (list): The list of channel names.
        eeg_format (string): The format of the EEG data (e.g. 'EDF', 'BDF', 'FIF').
        data_type (string): The type of data (e.g. 'eeg', 'meg', 'ieeg').
        filter_method (string): The method used for filtering the data.
        lowcut_freq (float): The low cutoff frequency for filtering.
        highcut_freq (float): The high cutoff frequency for filtering.
        sample_rate (float): The sample rate of the EEG data.
        ch2rm (list): The list of channel indices to be removed.
        path (string): The path to the directory where the HDF5 file should be saved.

    Outputs:
        None
    """
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, name + ".hdf")
    f = h5py.File(save_path, "w")
    dataset = f.create_dataset(name, data=data, compression="gzip", compression_opts=9)

    # add metadata
    dataset.attrs['data_length'] = data.shape[1]
    dataset.attrs['eeg_format'] = eeg_format
    dataset.attrs['data_type'] = data_type
    dataset.attrs['nchan'] = data.shape[0]
    dataset.attrs['ch_names'] = ch_names
    dataset.attrs['ch_removed'] = ch2rm
    dataset.attrs['sample_rate'] = sample_rate
    dataset.attrs['filter_method'] = filter_method
    dataset.attrs['lowcut_freq'] = lowcut_freq
    dataset.attrs['highcut_freq'] = highcut_freq

    f.close()
