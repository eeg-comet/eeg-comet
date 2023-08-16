
import numpy as np
import h5py
import os.path
import pandas as pd
from scipy.stats import mode
import math
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.stats import zscore
from itertools import groupby

from functions.utils.data_io import find_data, load_eegs, get_eeg_data
from functions.utils.substitude_maps_with_duration import substitude_maps_with_duration


def backfit_func(study_name, preprocessed_data_path, maps, method, filter_segments_option, remove_segments_less_than, micro_labels, save_path, extension, datatype, smooth_param):
    '''
    study_name: the name of the study which has been loaded
    preprocessed_data_path: the path of the preprocessed data
    maps: Gamma
    method: 'all' or 'peaks' (segmentation method)
    fs: sampling frequency
    filter_segments_option: 'replace' or 'remove' (option for handling segments with duration less than remove_segments_less_than)
    remove_segments_less_than: minimum duration of segments to remove or replace
    micro_labels: list of labels for each microstate
    save_path: path to save the segmentation data

    '''
    file_names = find_data(preprocessed_data_path, extension, "*")
    # Save group segmentation data
    hf_segmentation = h5py.File(os.path.join(save_path, study_name + '_segmentation.hdf'), 'w')
    group = hf_segmentation.create_group(study_name)
    counter = 1
    for filename in file_names:
        # hf = h5py.File(filename, "r")
        # data = hf[list(hf.keys())[0]]
        eeg = load_eegs(filename, extension, datatype)
        data = get_eeg_data(eeg, datatype)
        # if datatype == "epoched":
        #     for index in range(eeg.__len__()):
        #         if index == 0:
        #             data = np.squeeze(eeg[0].get_data())
        #         else:
        #             epoch = np.squeeze(eeg[index].get_data())
        #             data = np.append(data, epoch, axis=1)
        # else:
        #     data = eeg.get_data()
        #data = zscore(data, axis=1)
        filename = os.path.split(filename)[1].split('.')[0]
        print("\nBackfitting microstates to all time points", filename)

        if method == 'all':
            '''
            threshold = 0.7
            segmentation = np.zeros(data.shape[1], dtype=np.int32)
            for t in range(data.shape[1]):
                corr_append = np.zeros(maps.shape[0])
                for m in range(maps.shape[0]):
                    data_t = data[:, t]
                    data_t /= np.linalg.norm(data[:, t])
                    corr = abs(np.corrcoef(data_t, maps[m, :])[0, 1])
                    #corr = abs(cosine_sim(data[:, t], maps[m, :]))
                    corr_append[m] = corr
                high_corr = np.argmax(corr_append)
                if corr_append[high_corr] > threshold:
                    segmentation[t] = high_corr
                else:
                    segmentation[t] = -1
            '''
            # Calculate the correlation coefficient between each topography and each time point
            correlation_matrix = np.dot(maps, data) / np.sqrt(
                np.sum(maps ** 2, axis=1)[:, np.newaxis] * np.sum(data ** 2, axis=0))

            # Find the time point with the highest correlation coefficient for each topography
            segmentation = np.argmax(np.abs(correlation_matrix), axis=0).astype(int)

            # fill the mislabeled timepoints with nearby elements
            #segmentation = fill_missing_with_nearby(segmentation)

            '''
            segmentation = np.zeros(data.shape[1])
            for t in range(data.shape[1]):
                high_corr = 0
                corr_append = []
                for m in range(maps.shape[0]):
                    corr, _ = pearsonr(data[:, t], maps[m, :])
                    corr_append = np.append(corr_append, abs(corr))
                    high_corr = np.argmax(corr_append)
                segmentation[t] = high_corr
            '''
            #activation = np.array(maps).dot(data)
            #segmentation = np.argmax(np.abs(activation), axis=0)

            # Remove isolated segments
            if remove_segments_less_than:
                if filter_segments_option == 'replace':
                    print("Replacing segments with less than", str(remove_segments_less_than),
                          "ms in duration with the nearby dominant microstate")
                elif filter_segments_option == 'remove':
                    print("Removing segments with less than", str(remove_segments_less_than), "ms in duration")

                segmentation = substitude_maps_with_duration(segmentation,
                                                             remove_segments_less_than,
                                                             filter_segments_option,
                                                             data,
                                                             maps,
                                                             len(micro_labels),
                                                             smooth_param)
        elif method == 'peaks':
            gfp = np.std(data, axis=0)
            peaks, _ = find_peaks(gfp)
            troughs = [0]
            for p in range(len(peaks) - 1):
                min_arg = np.argmin((gfp[peaks[p]:peaks[p + 1]]))
                troughs = np.append(troughs, peaks[p] + min_arg)
            troughs = np.append(troughs, len(gfp))
            diff_troughs = np.diff(troughs)
            activation = maps.dot(data[:, peaks])
            segmentation_peaks = np.argmax(np.abs(activation), axis=0)
            segmentation = np.repeat(segmentation_peaks.astype(int), diff_troughs.astype(int))

        segmentation = segmentation + 1
        segmentation = list(map(int, segmentation))
        # Add Labels
        str_segmentation = list(map(str, segmentation))
        str_segmentation = np.char.replace(str_segmentation, str(0), "NaN")
        for m in range(1, len(micro_labels)+1):
            str_segmentation = np.char.replace(str_segmentation, str(m), micro_labels[m-1])

        # Encode the strings in a format h5py handles
        asciiList = [n.encode("ascii", "ignore") for n in str_segmentation]
        # Save data
        group.create_dataset(filename, (len(asciiList), 1), 'S10', asciiList, compression="gzip", compression_opts=9)

        print(100 * counter / len(file_names), "%")
        counter += 1
        #group.create_dataset(filename, data=str_segmentation, compression="gzip", compression_opts=9)


