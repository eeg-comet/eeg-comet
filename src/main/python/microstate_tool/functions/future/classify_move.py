#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:17:41 2021

@author: amin
"""

import os
from fnmatch import fnmatch
import shutil
from test_classifier import label_micromap

dataset_path = '/media/amin/Seagate Expansion Drive/AMIN/RS_EEG/RSEEG/img_clustering/5class_copy/'
output_path = '/media/amin/Seagate Expansion Drive/AMIN/RS_EEG/RSEEG/img_clustering/classify_new/'
extension = '*.png'

list_images = []
for path, subdirs, files in os.walk(dataset_path):
    for name in files:
        if fnmatch(name, extension):
            list_images.append(os.path.join(path, name))
      
for img in list_images:
    prediction = label_micromap(img)
    print(prediction)
    if prediction=='A':
        shutil.move(os.path.join(img), os.path.join(output_path, 'A'))
    elif prediction=='B':
        shutil.move(os.path.join(img), os.path.join(output_path, 'B'))
    elif prediction=='C':
        shutil.move(os.path.join(img), os.path.join(output_path, 'C'))
    elif prediction=='D':
        shutil.move(os.path.join(img), os.path.join(output_path, 'D'))
    elif prediction=='E':
        shutil.move(os.path.join(img), os.path.join(output_path, 'E'))
    elif prediction=='F':
        shutil.move(os.path.join(img), os.path.join(output_path, 'F'))
    
    