"""
Utility functions for adding labels to segmentations.

"""

import os.path
import numpy as np
import pandas as pd

from functions.data_utils.data_io import find_data, save_features


def add_segment_labels(segmentation_folder, fs, micro_labels, file_format):
    """
    Add microstate labels to segmentations in a given folder.
    """
    list_segmentations = find_data(segmentation_folder, file_format, '')
    for s in list_segmentations:
        filename = os.path.splitext(os.path.basename(s))[0]
        segmentation_df = pd.read_csv(s, header=0)
        segment = segmentation_df['segmentation'].astype(str)
        for m, label in enumerate(micro_labels):
            segment = segment.str.replace(str(m), label)
        time = np.arange(0, (1000 / fs) * len(segment), (1000 / fs))
        segmentation_df['time'] = time
        save_features(segmentation_df, f'raw_segmentation_{filename}', file_format, segmentation_folder)


def add_maps_labels(maps, micro_labels, file_format, eeg_info, maps_folder):
    """
    Add microstate labels to microstate maps.
    """
    maps_df = pd.DataFrame(maps.T, columns=micro_labels, index=eeg_info.ch_names)
    save_features(maps_df, 'microstate_maps', file_format, maps_folder)
