
import numpy as np
import h5py
import os.path
from scipy.signal import find_peaks

from functions.utils.substitude_maps_with_duration import substitude_maps_with_duration


def backfit_func(hf_group, maps, method, fs, min_duration, micro_labels, save_path):
    study_name = list(hf_group.keys())[0]
    file_names = list(hf_group[study_name].keys())
    # Save group segmentation data
    hf_segmentation = h5py.File(os.path.join(save_path, study_name + '_segmentation.hdf'), 'w')
    group = hf_segmentation.create_group(study_name)
    for filename in file_names:
        data = np.asarray(hf_group[study_name][filename][:])

        if method == 'all':
            activation = np.array(maps).dot(data)
            segmentation = np.argmax(np.abs(activation), axis=0)
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

        # Remove isolated segments
        if min_duration:
            min_duration = int(min_duration/(1000/fs))
            segmentation = substitude_maps_with_duration(segmentation, min_duration)

        # Add labels
        str_segmentation = list(map(str, segmentation))


        for m in range(len(micro_labels)):
            str_segmentation = np.char.replace(str_segmentation, str(m), micro_labels[m])

        #for i in range(len(micro_labels)):
        #    str_segmentation = np.char.replace(str_segmentation, str(i), micro_labels[i])

        # Encode the strings in a format h5py handles
        asciiList = [n.encode("ascii", "ignore") for n in str_segmentation]
        # Save data
        group.create_dataset(filename, (len(asciiList), 1), 'S10', asciiList, compression="gzip", compression_opts=9)

        #group.create_dataset(filename, data=str_segmentation, compression="gzip", compression_opts=9)

