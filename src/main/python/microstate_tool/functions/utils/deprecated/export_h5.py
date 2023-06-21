
import os.path
import h5py


def export_h5(data, filename, ch_names, eeg_format, data_type, filter_method,
              lowcut_freq, highcut_freq, sample_rate, ch2rm, path):
    '''
    data: Input data matrix
    filename: Name of the file to be exported
    ch_names: List of channel names
    eeg_format: EEG data format
    data_type: Data type of the exported data
    filter_method: Method used for data filtering
    lowcut_freq: Low cutoff frequency for filtering
    highcut_freq: High cutoff frequency for filtering
    sample_rate: Sampling rate of the data
    ch2rm: Channels to be removed from the data
    path: Path to save the exported file
    '''
    
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, name + ".hdf")
    f = h5py.File(save_path, "w")
    dataset = f.create_dataset(name, data=data, compression="gzip", compression_opts=9)

    # add metadata
    dataset.attrs['data_length'] = data.shape[1]
    dataset.attrs['eeg_format'] = eeg_format
    dataset.attrs['data_type'] = data_type
    dataset.attrs['nchan'] = data.shape[0]
    dataset.attrs['ch_names'] = ch_names
    dataset.attrs['ch_removed'] = ch2rm
    dataset.attrs['sample_rate'] = sample_rate
    dataset.attrs['filter_method'] = filter_method
    dataset.attrs['lowcut_freq'] = lowcut_freq
    dataset.attrs['highcut_freq'] = highcut_freq

    f.close()
