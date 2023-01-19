
import numpy as np
import h5py
import os.path
from scipy.signal import find_peaks

from functions.utils.substitude_maps_with_duration import substitude_maps_with_duration


def backfit_func(hf_group, maps, method, fs, filter_segments_option, remove_segments_less_than, micro_labels, save_path):
    study_name = list(hf_group.keys())[0]
    file_names = list(hf_group[study_name].keys())
    # Save group segmentation data
    hf_segmentation = h5py.File(os.path.join(save_path, study_name + '_segmentation.hdf'), 'w')
    group = hf_segmentation.create_group(study_name)
    counter = 1
    for filename in file_names:
        data = np.asarray(hf_group[study_name][filename][:])
        print("\nSegmenting", filename)

        if method == 'all':
            activation = np.array(maps).dot(data)
            segmentation = np.argmax(np.abs(activation), axis=0)

            # Remove isolated segments
            if remove_segments_less_than:
                if filter_segments_option == 'replace':
                    print("Replacing segments with less than", str(remove_segments_less_than),
                          "ms in duration with the previous dominant microstate")
                elif filter_segments_option == 'remove':
                    print("Removing segments with less than", str(remove_segments_less_than), "ms in duration")
                segmentation = substitude_maps_with_duration(segmentation,
                                                             fs,
                                                             remove_segments_less_than,
                                                             filter_segments_option)
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

