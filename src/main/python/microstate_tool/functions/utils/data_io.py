import pickle
import os.path
import h5py
from scipy.stats import zscore
import mne
from fnmatch import fnmatch
from configparser import ConfigParser
import numpy as np


def import_hdf_data(hdf_data_path):
    hf = h5py.File(hdf_data_path, "r")
    study_name = list(hf.keys())[0]
    dset = hf[study_name]
    dset = np.asarray(dset)
    dset = zscore(dset, axis=1)
    return dset


def save_eeg_info(eeg_info_path, eeg_info):
    with open(eeg_info_path, 'wb') as f:
        pickle.dump(eeg_info, f)


def load_eeg_info(eeg_info_path):
    with open(eeg_info_path, 'rb') as f:
        eeg_info = pickle.load(f)
    return eeg_info


def find_data(input_folder, extension, pattern):
    '''
    input_folder: Path to the folder where data files are located
    extension: File extension to search for (e.g., ".hdf")
    pattern: Pattern to match in file names (e.g., "*")

    Recursively searches the input folder for files with the specified extension
    and matching the given pattern.

    Returns a list of file paths matching the search criteria.
    '''
    list_data = []
    for path, subdirs, files in os.walk(input_folder):
        for name in files:
            if fnmatch(name, pattern+extension):
                list_data.append(os.path.join(path, name))
    return list_data


def load_eegs(filename, eeg_format, datatype, channel_location_dir='', chan2rm=[]):
    if datatype == 'raw':

        # Load the eeg file
        if eeg_format == ".vhdr":
            eeg = mne.io.read_raw_brainvision(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".edf":
            eeg = mne.io.read_raw_edf(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".bdf":
            eeg = mne.io.read_raw_bdf(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".gdf":
            eeg = mne.io.read_raw_gdf(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".cnt":
            eeg = mne.io.read_raw_cnt(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".egi" or eeg_format == ".mff":
            eeg = mne.io.read_raw_egi(filename, preload=True, verbose='CRITICAL')            
        elif eeg_format == ".set":
            eeg = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
            # mne.export.export_raw('/Users/bottlecap/Downloads/eeg.set', eeg, fmt='eeglab', verbose='CRITICAL')
        elif eeg_format == ".data":
            eeg = mne.io.read_raw_nicolet(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".nxe":
            eeg = mne.io.read_raw_eximia(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".lay":
            eeg = mne.io.read_raw_persyst(filename, preload=True, verbose='CRITICAL')
        elif eeg_format == ".eeg":
            eeg = mne.io.read_raw_nihon(filename, preload=True, verbose='CRITICAL')
    elif datatype == 'epoched':
        if eeg_format == ".set":
            eeg = mne.io.read_epochs_eeglab(filename, verbose='CRITICAL')

    # Load the eeg channel location
    if channel_location_dir:
        montage = mne.channels.read_custom_montage(channel_location_dir)
        eeg.set_montage(montage)

    # Pick channels
    eeg = eeg.pick_types(meg=False, eeg=True, eog=False,
                         exclude=chan2rm, verbose='CRITICAL')

    # Apply an average reference
    eeg = eeg.set_eeg_reference('average')
    return eeg

# def dump_eegs(filename, eeg_format, datatype, eeg):

def export_eegs(eeg, save_path, extension, datatype):
    print('*'*20, 'save')
    avaliable_extensions = ['.vhdr', '.set', '.edf']
    if not extension in avaliable_extensions:
        extension = '.set'
    if datatype == 'raw':
        mne.export.export_raw(save_path+extension, eeg, fmt='auto', overwrite=True)
    elif datatype == 'epoched':
        mne.export.export_epochs(save_path+extension, eeg, fmt='auto', overwrite=True)

def get_eeg_data(eeg, datatype):
    if datatype == "epoched":
        # TODO: try to use reshape instead of append
        for index in range(eeg.__len__()):
            if index == 0:
                data = np.squeeze(eeg[0].get_data())
            else:
                epoch = np.squeeze(eeg[index].get_data())
                data = np.append(data, epoch, axis=1)
    else:
        data = eeg.get_data()
    return data

def initialize_config(config_path, config):
    # Create sections
    config['progress'] = {}
    config['study info'] = {}
    config['preprocessing settings'] = {}
    config['preprocessing results'] = {}
    config['clustering settings'] = {}
    config['clustering results'] = {}
    config['backfitting settings'] = {}
    config['feature extraction settings'] = {}
    config['feature visualization groups'] = {}
    config['source localization settings'] = {}
    # Initialize progress
    str_false = "False"
    config['progress']['done_preprocessing'] = str_false
    config['progress']['done_clustering'] = str_false
    config['progress']['done_labeling_microstates'] = str_false
    config['progress']['done_backfitting'] = str_false
    config['progress']['done_extracting_features'] = str_false
    config['progress']['done_extracting_microsegments'] = str_false
    config['progress']['done_source_localization'] = str_false
    # Write to config
    save_config(config_path, config)


def load_config(config_path):
    config = ConfigParser()
    config.read(config_path)
    return config


def save_config(config_path, config):
    with open(config_path, 'w+') as configfile:
        config.write(configfile)
    return config
    

def save_features(df, filename, file_format, path):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, filename + file_format)
    if file_format == '.csv':
        df.to_csv(save_path, header=True, index=False)
    elif file_format == '.pkl':
        df.to_pickle(save_path)
    elif file_format == '.hdf':
        df.to_hdf(save_path, key='df', mode='w')
    elif file_format == '.json':
        df.to_json(save_path)
        

# export_h5 seems deprecated!!!

# def export_h5(data, filename, ch_names, eeg_format, data_type, filter_method,
#               lowcut_freq, highcut_freq, sample_rate, ch2rm, path):
#     '''
#     data: Input data matrix
#     filename: Name of the file to be exported
#     ch_names: List of channel names
#     eeg_format: EEG data format
#     data_type: Data type of the exported data
#     filter_method: Method used for data filtering
#     lowcut_freq: Low cutoff frequency for filtering
#     highcut_freq: High cutoff frequency for filtering
#     sample_rate: Sampling rate of the data
#     ch2rm: Channels to be removed from the data
#     path: Path to save the exported file
#     '''
    
#     name = os.path.basename(filename)
#     name = os.path.splitext(name)[0]
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_path = os.path.join(path, name + ".hdf")
#     f = h5py.File(save_path, "w")
#     dataset = f.create_dataset(name, data=data, compression="gzip", compression_opts=9)

#     # add metadata
#     dataset.attrs['data_length'] = data.shape[1]
#     dataset.attrs['eeg_format'] = eeg_format
#     dataset.attrs['data_type'] = data_type
#     dataset.attrs['nchan'] = data.shape[0]
#     dataset.attrs['ch_names'] = ch_names
#     dataset.attrs['ch_removed'] = ch2rm
#     dataset.attrs['sample_rate'] = sample_rate
#     dataset.attrs['filter_method'] = filter_method
#     dataset.attrs['lowcut_freq'] = lowcut_freq
#     dataset.attrs['highcut_freq'] = highcut_freq

#     f.close()

