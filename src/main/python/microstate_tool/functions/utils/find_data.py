#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:01:17 2021
Last Modified: April 18th, 2023
Description: Find data within a folder with a matching extension and string pattern.

Inputs: input_folder (string): Input folder
        extension (string): Desired file extension
        pattern (string): Desired string pattern

Outputs: list_data (list of strings): List of filenames

@author: Amin Kabir, Raaj Chatterjee, Faranak Farzan
eBrain Lab, 2023
"""

import os
from fnmatch import fnmatch


def find_data(input_folder, extension, pattern):
    list_data = []
    for path, subdirs, files in os.walk(input_folder):
        for name in files:
            if fnmatch(name, pattern+extension):
                list_data.append(os.path.join(path, name))
    return list_data

