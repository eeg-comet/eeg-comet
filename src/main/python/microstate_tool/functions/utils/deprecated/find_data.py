#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:01:17 2021

@author: amin
"""

import os
from fnmatch import fnmatch


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

