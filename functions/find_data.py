#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:01:17 2021

@author: amin
"""

import os
from fnmatch import fnmatch

def find_eeg(input_folder, extension, pattern):
    list_eegs = []
    for path, subdirs, files in os.walk(input_folder):
        for name in files:
            if fnmatch(name, pattern+extension):
                list_eegs.append(os.path.join(path, name))
    return list_eegs