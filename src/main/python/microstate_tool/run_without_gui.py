# Running the microstate pipeline without loading the GUI

from configparser import ConfigParser
import os.path
import numpy as np
import pandas as pd

from functions import find_data, preprocess
from functions.concatenate_data import concatenate_files
from functions.clustering_functions import pre_clustering, initialize_centers, eegInfo
from functions.clustering_functions import number_of_clusters, clustering_func, clustering_minibatch
from functions.extract_features_functions import save_raw_results, extract_features, save_features, substitude_maps_with_duration


# Input: 'settings_log.ini'
settings_path = os.path.join('C:\\Users\\amin_\\Documents\\GitHub\\output_test\\',
                             'settings_log.ini')

# Load Input Settings
config = ConfigParser()
config.read(settings_path)
study_name = config.get('input_settings', 'study_name')
input_folder = config.get('input_settings', 'input_folder')
pattern = config.get('input_settings', 'pattern')
save_dir = config.get('input_settings', 'save_folder')
extension = config.get('input_settings', 'data_extension')
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
smoothing_kernel_size = int(config.get('clustering_settings', 'smoothing_kernel_size'))
tolerance = float(config.get('clustering_settings', 'tolerance'))
concatenate_data = config.get('clustering_settings', 'concatenate_data')
number_of_repeats = int(config.get('clustering_settings', 'number_of_repeats'))
# Load Features Settings
features = config.get('features_settings', 'features')
remove_segments_less_than = int(config.get('features_settings', 'remove_segments_less_than'))
save_raw_segmentation = config.get('features_settings', 'save_raw_segmentation')
save_microstate_maps = config.get('features_settings', 'save_microstate_maps')
save_transition_matrices = config.get('features_settings', 'save_transition_matrices')
output_format = config.get('features_settings', 'output_format')

# Find raw EEG files inside the input folder
list_eegs = find_data.find_eeg(input_folder, extension, pattern)
# Preprocessing the raw data
save_preprocessed_path = os.path.join(save_dir, 'preprocessed_data')

length_all_data = []
for file in list_eegs:
    progress, length_data = preprocess.preprocess_eegs(file, list_eegs,
                                extension,
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
config_file = os.path.join(save_dir, 'data_log.ini')
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
DATA, N_CHANNELS, FILENAMES = concatenate_files(save_dir)
# Smoothing the input data
MAPS, PEAKS = pre_clustering(DATA, sample_rate, smoothing_kernel_size)
# Finding the initial maps
INITIAL_CENTERS = initialize_centers(
                DATA,
                MAPS,
                PEAKS,
                number_of_maps,
                initializer)
# Performing the clustering
best_maps, final_segmentation, gev = clustering_func(
                DATA,
                N_CHANNELS,
                MAPS,
                clustering_method,
                number_of_maps,
                INITIAL_CENTERS,
                smoothing_kernel_size,
                number_of_repeats,
                tolerance,
                option)
config.add_section('clustering_results')
config.set('clustering_results', 'gev', str(gev))
with open(config_file, 'w+') as f:
    config.write(f)
# Saving the raw results
save_raw_path = os.path.join(save_dir, 'raw_features')
if not os.path.exists(save_raw_path):
    os.makedirs(save_raw_path)
# Save Segmentation
time = np.arange(0, (1000 / int(sample_rate)) * len(final_segmentation), (1000 / int(sample_rate)))
final_segmentation = substitude_maps_with_duration(final_segmentation, int(remove_segments_less_than/(1000/int(sample_rate))))
segmentation_df = pd.DataFrame(final_segmentation,
                               columns=['segmentation'],
                               index=time)
save_name = os.path.join(save_raw_path, 'raw_segmentation')
segmentation_df.to_csv(save_name + '.csv')

micro_labels = ['MAP'+str(i) for i in range(1, int(number_of_maps)+1)]
#micro_labels = ['M1', 'M2', 'M3', 'M4', 'M5']

save_raw_results(FILENAMES, length_all_data, sample_rate,
                    save_raw_segmentation, final_segmentation,
                    save_microstate_maps, best_maps,
                    micro_labels,
                    save_transition_matrices,
                    save_dir,
                    output_format, save_raw_path)


# Save Features
save_features_path = os.path.join(save_dir, 'extracted_features')
if not os.path.exists(save_features_path):
    os.makedirs(save_features_path)

extracted_features_df = extract_features(FILENAMES, length_all_data,
                                         final_segmentation, best_maps,
                                         sample_rate, features)
save_features(extracted_features_df, output_format, save_features_path)