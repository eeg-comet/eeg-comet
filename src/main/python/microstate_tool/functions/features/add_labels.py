
import os.path
import numpy as np
import pandas as pd

from functions.utils.data_io import find_data
from functions.utils.save_features import save_features


def add_segment_labels(segmentation_folder, fs, micro_labels, file_format):
    list_segmentations = find_data(segmentation_folder, file_format, '')
    for s in list_segmentations:
        filename = os.path.splitext(os.path.basename(s))[0]
        segment = pd.read_csv(s, header=0)['segmentation'].to_list()
        for m in range(len(micro_labels)):
            segment = np.char.replace(segment, str(m), micro_labels[m])
        time = np.arange(0, (1000 / fs) * len(segment), (1000 / fs))
        segmentation_df = pd.DataFrame({'time': time, 'segmentation': segment})
        save_features(segmentation_df, 'raw_segmentation_' + filename[0], file_format, segmentation_folder)

def add_maps_labels(maps, micro_labels, file_format, eeg_info, maps_folder):
    maps_df = pd.DataFrame(maps.T, columns=micro_labels, index=eeg_info.ch_names)
    save_features(maps_df, 'microstate_maps', file_format, maps_folder)