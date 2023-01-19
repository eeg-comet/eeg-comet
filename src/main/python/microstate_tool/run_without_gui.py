# Running the microstate pipeline without loading the GUI
import argparse
from configparser import ConfigParser
import os.path
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage.filters import gaussian_filter1d

from src.main.python.microstate_tool.functions.utils import find_data
from functions.clustering_functions import backfit_func
from src.main.python.microstate_tool.functions.features.extract_features_functions import save_raw_results, extract_features, save_features

def main(args):
    # Input: 'settings_log.ini'
    #settings_path = args.setting_path
    settings_folder = 'C://Users//amin_//Documents//GitHub//output_test//test_new//'
    #settings_folder = '//home//aminka//scratch//rs_eeg//RS_MICROSTATES_RESULTS//RS_MICROSTATES_EC_5MAPS_2_20Hz'
    settings_path = os.path.join(settings_folder, 'settings_log.ini')

    print(settings_path)
    # Load Input Settings
    config = ConfigParser()
    config.read(settings_path)
    study_name = config.get('input_settings', 'study_name')
    input_folder = config.get('input_settings', 'input_folder')
    pattern = config.get('input_settings', 'pattern')
    save_folder = config.get('input_settings', 'save_folder')
    data_extension = config.get('input_settings', 'data_extension')
    data_type = config.get('input_settings', 'data_type')
    filter_data = config.get('input_settings', 'filter_data')
    filter_method = config.get('input_settings', 'filter_method')
    lowcut_freq = int(config.get('input_settings', 'lowcut_freq'))
    highcut_freq = int(config.get('input_settings', 'highcut_freq'))
    downsample_data = config.get('input_settings', 'downsample_data')
    sample_rate = int(config.get('input_settings', 'sample_rate'))

    # Load Clustering Settings
    clustering_method = config.get('clustering_settings', 'clustering_method')
    option = config.get('clustering_settings', 'option')
    choose_number_of_maps = config.get('clustering_settings', 'choose_number_of_maps')
    number_of_maps = int(config.get('clustering_settings', 'number_of_maps'))
    initializer = config.get('clustering_settings', 'initializer')
    smoothing_gfp = config.get('clustering_settings', 'smoothing_gfp')
    min_distance_size = int(config.get('clustering_settings', 'min_distance_size'))
    tolerance = float(config.get('clustering_settings', 'tolerance'))
    concatenate_data = config.get('clustering_settings', 'concatenate_data')
    number_of_repeats = int(config.get('clustering_settings', 'number_of_repeats'))

    # Load Features Settings
    features = config.get('features_settings', 'features')
    save_raw_segmentation = config.get('features_settings', 'save_raw_segmentation')
    save_microstate_maps = config.get('features_settings', 'save_microstate_maps')
    save_transition_matrices = config.get('features_settings', 'save_transition_matrices')
    output_format = config.get('features_settings', 'output_format')

    # Load Backfitting Settings
    backfit_to = config.get('backfitting_settings', 'backfit_to')
    remove_segments_less_than = config.get('backfitting_settings', 'remove_segments_less_than')
    if remove_segments_less_than:
        remove_segments_less_than = int(float(remove_segments_less_than))

    ###
    config = ConfigParser()
    config_file = os.path.join(save_folder, 'data_log.ini')
    config.read(config_file)

    FILENAMES = config.get('input_data', 'list_eegs')
    FILENAMES = FILENAMES.split(",")
    length_all_data = config.get('input_data', 'length_data')
    length_all_data = length_all_data.split(",")
    length_all_data = list(map(float, length_all_data))
    length_all_data = list(map(int, length_all_data))

    # Load maps
    micro_labels = ['B', 'E', 'C', 'A', 'D']
    save_raw_path = os.path.join(save_folder, 'raw_features')
    raw_results_path = os.path.join(save_folder, 'raw_features')
    final_maps_path = os.path.join(raw_results_path, 'microstate_maps.csv')
    final_segmentation_path = os.path.join(raw_results_path, 'raw_segmentation.csv')

    final_maps = pd.read_csv(final_maps_path)
    final_maps = final_maps.iloc[:, 1:]
    final_maps = final_maps.values.T

    for dirpath, dirnames, filenames in os.walk(save_folder):
        for filename in [f for f in filenames if f.startswith("EEG_INFO")]:
            eegInfo_path = os.path.join(dirpath, filename)

    conatenated_data_path = os.path.join(save_folder, 'catdata.h5')
    if os.path.exists(conatenated_data_path):
        with h5py.File(conatenated_data_path, "r") as f:
            a_group_key = list(f.keys())[0]
            data = list(f[a_group_key])
        CONCAT_DATA = np.asarray(data)
    ###

    '''
    # Find raw EEG files inside the input folder
    list_eegs = find_data.find_eeg(input_folder, data_extension, pattern)
    # Preprocessing the raw data
    save_preprocessed_path = os.path.join(save_folder, 'preprocessed_data')

    length_all_data = []
    for file in list_eegs:
        progress, length_data = preprocess.preprocess_eegs(file, list_eegs,
                                    data_extension,
                                    data_type,
                                    filter_data,
                                    filter_method,
                                    lowcut_freq,
                                    highcut_freq,
                                    downsample_data,
                                    sample_rate,
                                    save_preprocessed_path)
        length_all_data = np.append(length_all_data, int(length_data))
        print("progress: ", progress)

    # Save data log
    list_eegs_save = ','.join(map(str, list_eegs))
    config = ConfigParser()
    config_file = os.path.join(save_folder, 'data_log.ini')
    if os.path.isfile(config_file):
        os.remove(config_file)
    config.read(config_file)
    config.add_section('input_data')
    config.set('input_data', 'list_eegs', list_eegs_save)
    length_data_save = ','.join(map(str, length_all_data))
    config.set('input_data', 'length_data', length_data_save)
    with open(config_file, 'w+') as f:
        config.write(f)

    # Concatenating the preprocessed data
    CONCAT_DATA, N_CHANNELS, FILENAMES = concatenate_files(save_folder)
    # Smoothing the input data
    MAPS, PEAKS = pre_clustering(CONCAT_DATA, min_distance_size)
    # Finding the initial maps
    INITIAL_CENTERS = initialize_centers(
                    CONCAT_DATA,
                    MAPS,
                    PEAKS,
                    number_of_maps,
                    initializer)

    # Performing the clustering
    final_maps, gev = clustering_func(
        CONCAT_DATA,
        N_CHANNELS,
        clustering_method,
        number_of_maps,
        INITIAL_CENTERS,
        min_distance_size,
        number_of_repeats,
        tolerance,
        option)

    # Saving the raw results
    save_raw_path = os.path.join(save_folder, 'raw_features')
    for dirpath, dirnames, filenames in os.walk(save_folder):
        for filename in [f for f in filenames if f.startswith("EEG_INFO")]:
            eegInfo_path = os.path.join(dirpath, filename)
    with open(eegInfo_path, 'rb') as f:
        eeg_info = pickle.load(f)
    #if micro_labels != []:
    #    maps_df = pd.DataFrame(final_maps.T, columns=micro_labels, index=eeg_info.ch_names)
    #else:
    maps_df = pd.DataFrame(final_maps.T, index=eeg_info.ch_names)
    save_features(maps_df, 'microstate_maps', output_format, save_raw_path)

    config.add_section('clustering_results')
    config.set('clustering_results', 'gev', str(gev))
    with open(config_file, 'w+') as f:
        config.write(f)
    '''

    # Performing the backfitting
    print('Backfitting ...')
    CONCAT_DATA = gaussian_filter1d(CONCAT_DATA, sigma=2)
    final_segmentation = backfit_func(CONCAT_DATA,
                                      final_maps,
                                      backfit_to,
                                      remove_segments_less_than)

    # Save Segmentation
    time = np.arange(0, (1000 / int(sample_rate)) * len(final_segmentation), (1000 / int(sample_rate)))
    segmentation_df = pd.DataFrame(final_segmentation,
                                   columns=['segmentation'],
                                   index=time)
    raw_segmentation_folder = os.path.join(save_raw_path, 'raw_segmentation')
    save_features(segmentation_df, 'raw_segmentation', output_format, raw_segmentation_folder)
    done_backfitting = True
    print('done')

    '''
    config.add_section('clustering_results')
    config.set('clustering_results', 'gev', str(gev))
    with open(config_file, 'w+') as f:
        config.write(f)
    '''
    
    #micro_labels = ['MAP'+str(i) for i in range(1, int(number_of_maps)+1)]
    #micro_labels = ['B', 'E', 'C', 'A', 'D']

    final_segmentation = list(map(str, final_segmentation))
    for i in range(len(micro_labels)):
        final_segmentation = np.char.replace(final_segmentation, str(i), micro_labels[i])

    save_raw_results(FILENAMES, length_all_data, sample_rate,
                        save_raw_segmentation, final_segmentation,
                        save_microstate_maps, final_maps,
                        micro_labels,
                        save_transition_matrices,
                        save_folder,
                        output_format, save_raw_path)


    # Save Features
    save_features_path = os.path.join(save_folder, 'extracted_features')
    if not os.path.exists(save_features_path):
        os.makedirs(save_features_path)

    listofh5files = find_data.find_eeg(os.path.join(save_folder, 'preprocessed_data'), '.h5', '*')
    extracted_features_df = extract_features(FILENAMES, length_all_data, listofh5files,
                                             final_segmentation, final_maps,
                                             sample_rate, features)
    save_features(extracted_features_df, 'extracted_features', output_format, save_features_path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_path', type=str, default='settings_log.ini')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
