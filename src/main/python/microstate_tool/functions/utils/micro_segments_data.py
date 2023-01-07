
import os.path
import numpy as np
import h5py


def micro_segments_data(hf_data_path, hf_segmentation_path, n_chan, micro_labels, save_path):
    # Load data and segment files
    hf_data = h5py.File(hf_data_path, 'r')
    hf_segmentation = h5py.File(hf_segmentation_path, 'r')
    study_name = list(hf_data.keys())[0]
    file_names = list(hf_data[study_name].keys())
    for micro_map in micro_labels:
        print("Extracting segments matching microstate ", micro_map)
        data_map_concatenated = np.empty((n_chan,0))
        for filename in file_names:
            print(study_name, filename)
            data = np.asarray(hf_data[study_name][filename][:])
            segment = np.asarray(hf_segmentation[study_name][filename][:])
            segment = [" ".join(item) for item in segment.astype('U10')]
            segment = np.asarray(segment)
            ind_map = np.where(segment == micro_map)[0]
            data_map = data[:, ind_map]
            data_map_concatenated = np.concatenate((data_map_concatenated, data_map), axis=1)
        # save
        output_path = os.path.join(save_path, micro_map + ".hdf")
        f = h5py.File(output_path, "w")
        f.create_dataset(micro_map, data=data_map_concatenated, compression="gzip", compression_opts=9)

