#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 09:29:23 2021

@author: amin
"""

import os
import numpy as np
from fnmatch import fnmatch
import h5py
from scipy.stats import zscore
from configparser import ConfigParser

from functions.utils.load_save_config import load_config, save_config


def concatenate_files(study_name, outputfolder):

    # load data log
    config_file = os.path.join(outputfolder, 'log.ini')
    config = load_config(config_file)
    if not config.has_option('clustering settings', 'concat_data_available'):
        counter = 0
        eeglist = []
        extension = "*.h5"
        for path, subdirs, files in os.walk(os.path.join(outputfolder, 'preprocessed_data')):
            for name in files:
                if fnmatch(name, extension):
                    eeglist.append(os.path.join(path, name))

        hf = h5py.File(os.path.join(outputfolder, study_name + '_data.hdf'), 'w')
        group = hf.create_group(study_name)

        for filename in eeglist:
            print(100 * counter / len(eeglist))
            print('\nLoading EEG Files ... ', filename)

            f = h5py.File(filename, 'r')
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            dataset = f[name]
            data = np.asarray(dataset[:])

            data_length = dataset.attrs['data_length']
            eeg_format = dataset.attrs['eeg_format']
            data_type = dataset.attrs['data_type']
            nchan = dataset.attrs['nchan']
            ch_names = dataset.attrs['ch_names']
            ch_removed = dataset.attrs['ch_removed']
            sample_rate = dataset.attrs['sample_rate']
            filter_method = dataset.attrs['filter_method']
            lowcut_freq = dataset.attrs['lowcut_freq']
            highcut_freq = dataset.attrs['highcut_freq']

            f.close()

            group.create_dataset(name, data=data, compression="gzip", compression_opts=9)

            hf.attrs['data_length'] = data_length
            hf.attrs['eeg_format'] = eeg_format
            hf.attrs['data_type'] = data_type
            hf.attrs['nchan'] = nchan
            hf.attrs['ch_names'] = ch_names
            hf.attrs['ch_removed'] = ch_removed
            hf.attrs['sample_rate'] = sample_rate
            hf.attrs['filter_method'] = filter_method
            hf.attrs['lowcut_freq'] = lowcut_freq
            hf.attrs['highcut_freq'] = highcut_freq

        print("\nSaving the concatenated data ...")

        config['clustering settings']['concat_data_available'] = str(True)
        save_config(config_file, config)

    else:
        preprocessed_data_dir = os.path.join(outputfolder, study_name + '_data.hdf')
        hf = h5py.File(preprocessed_data_dir, 'r')

        # load config
        #config = ConfigParser()
        #config_file = os.path.join(outputfolder, 'log.ini')
        #config.read(config_file)
        #filenames = config.get('input_data', 'list_eegs')

    dset = hf[study_name]
    counter = 0
    for k in list(dset.keys()):
        dataset_k = list(dset[k])
        dataset_k = np.asarray(dataset_k)
        if counter == 0:
            filenames = k
            catdata = dataset_k
        else:
            filenames = np.append(filenames, k)
            catdata = np.append(catdata, dataset_k, axis=1)
        counter += 1

    catdata = zscore(catdata, axis=1)
    nchan = hf.attrs['nchan']
    #nchan = catdata.shape[0]
    return hf, catdata, nchan, filenames


def get_file_info(outputfolder):

    # load data log
    config_file = os.path.join(outputfolder, 'log.ini')
    config = load_config(config_file)
    if not config.has_option('clustering settings', 'concat_data_available'):
        counter = 0
        eeglist = []
        extension = "*.h5"
        for path, subdirs, files in os.walk(os.path.join(outputfolder, 'preprocessed_data')):
            for name in files:
                if fnmatch(name, extension):
                    eeglist.append(os.path.join(path, name))

        for filename in eeglist:
            print(100 * counter / len(eeglist))
            print('\nLoading EEG Files ... ', filename)
            # Load the example MNE data
            with h5py.File(filename, "r") as f:
                # List all groups
                print("Keys: %s" % f.keys())
                a_group_key = list(f.keys())[0]
                # Get the data
                data = list(f[a_group_key])

            data_tmp = np.asarray(data)
            print(data_tmp.shape)
            data_len = data_tmp.shape[1]

            if counter == 0:
                filenames = filename
                catdata = data_tmp
            else:
                filenames = np.append(filenames, filename)
                catdata = np.append(catdata, data_tmp, axis=1)
            counter += 1

        print("\nSaving the concatenated data ...")
        with h5py.File(os.path.join(outputfolder, 'catdata.h5'), 'w') as f:
            f.create_dataset('catdata', data=catdata)
        config['clustering settings']['concat_data_available'] = str(True)
        save_config(config_file, config)

    else:
        catdata_dir = os.path.join(outputfolder, 'catdata.h5')
        with h5py.File(catdata_dir, "r") as f:
            a_group_key = list(f.keys())[0]
            catdata = list(f[a_group_key])
        catdata = np.asarray(catdata)
        # load config
        filenames = config['study info']['input_filenames']
    nchan = catdata.shape[0]
    return catdata, nchan, filenames