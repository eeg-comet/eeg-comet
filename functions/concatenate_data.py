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

#input_folder="/home/amin/Encfs/TMSEEG_DATA/EEG-Microstate-Feature-Extraction/test_data/save_folder/"

def concatenate_files(folder):
    counter = 0
    
    eeglist = []
    extension="*.h5"
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if fnmatch(name, extension):
                eeglist.append(os.path.join(path, name))
    
    for filename in eeglist:
        print(100*counter/len(eeglist))
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
    nchan = catdata.shape[0]
    print("\nSaving the concatenated data ...")
    with h5py.File(os.path.join(folder, 'catdata.h5'),'w') as f:
            f.create_dataset('catdata', data=catdata)
    return catdata, nchan, filenames, data_length

#INPUT_DATA, N_CHANNELS, FILENAMES, LENGTH_DATA = concatenate_files(input_folder)

'''
print(INPUT_DATA.shape)
# Load Variable
with open(os.path.join(input_folder,'EEG_INFO.pickle'), 'rb') as f:
    EEG_INFO = pickle.load(f)
n_channels = EEG_INFO['nchan']
Fs = 250
'''

