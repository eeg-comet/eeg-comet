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
from configparser import ConfigParser


def concatenate_files(outputfolder):

    # load data log
    config = ConfigParser()
    config_file = os.path.join(outputfolder, 'data_log.ini')
    config.read(config_file)
    if not config.has_option('input_data', 'concat_data_available'):
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
                data_length = data_len
            else:
                filenames = np.append(filenames, filename)
                catdata = np.append(catdata, data_tmp, axis=1)
                data_length = np.append(data_length, data_len)
            counter += 1

        print("\nSaving the concatenated data ...")
        with h5py.File(os.path.join(outputfolder, 'catdata.h5'), 'w') as f:
            f.create_dataset('catdata', data=catdata)
        config.set('input_data', 'concat_data_available', 'True')
        with open(config_file, 'w+') as f:
            config.write(f)

    else:
        catdata_dir = os.path.join(outputfolder, 'catdata.h5')
        with h5py.File(catdata_dir, "r") as f:
            a_group_key = list(f.keys())[0]
            catdata = list(f[a_group_key])
        catdata = np.asarray(catdata)
        # load config
        config = ConfigParser()
        config_file = os.path.join(outputfolder, 'data_log.ini')
        config.read(config_file)
        filenames = config.get('input_data', 'list_eegs')
        data_length = config.get('input_data', 'length_data')
    nchan = catdata.shape[0]
    return catdata, nchan, filenames, data_length
