
import os.path
import numpy as np
import pandas as pd
from configparser import ConfigParser

from functions import find_data, preprocess
from functions.concatenate_data import concatenate_files
from functions.clustering_functions import pre_clustering, initialize_centers, eegInfo
from functions.clustering_functions import number_of_clusters, clustering_func, clustering_minibatch
from functions.extract_features_functions import save_raw_results, extract_features, save_features, substitude_maps_with_duration

import nltk.cluster as clustering

settings_folder = 'C://Users//amin_//Documents//GitHub//new_clustering_test//'
settings_path = os.path.join(settings_folder, 'settings_log.ini')


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
MAPS, PEAKS = pre_clustering(DATA, sample_rate, 20)
print(MAPS.shape)
np.save('C://Users//amin_//Documents\GitHub//new_clustering_test//MAPS', MAPS)


NUM_CLUSTERS = 5
data = MAPS

def cosine_distance(u, v):
    """
    Returns 1 minus the cosine of the angle between vectors v and u. This is
    equal to ``1 - (u.v / |u||v|)``.
    """
    return 1 - abs(np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v))))

kclusterer = clustering.kmeans.KMeansClusterer(NUM_CLUSTERS, distance=cosine_distance, repeats=5)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)

print(assigned_clusters)
print(kclusterer.means())
np.save('C://Users//amin_//Documents//GitHub//new_clustering_test//topo//MAPS_MEAN', kclusterer.means())