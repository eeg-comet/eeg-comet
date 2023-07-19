from functions.utils.data_io import find_data, load_eegs, save_eeg_info, initialize_config, load_config, load_eeg_info
import os
from functions import preprocess
import h5py
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from functions.clustering_functions import number_of_clusters, clustering_func
from functions.utils.backfit_func import backfit_func
from functions.features.extract_features_functions import extract_segments, save_segmentation_results,\
	save_transitions, save_raw_results, extract_features, transition_matrix, save_features, extract_dynamic_features
import matplotlib.pyplot as plt
import mne
from functions.features.source_localization_tess import run_source_localization #, visualize_sources
import pickle


class ToolBox:
	def __init__(self, config):
		# base
		self.config = config
		self.study_name = config['base']['study_name']
		self.input_folder = config['base']['input_folder']
		self.channel_location_dir = config['base']['channel_location_dir']
		self.output_folder = config['base']['output_folder']

		self.save_dir = os.path.join(self.output_folder, self.study_name)
		assert self.study_name != "", "study name cannot be empty"
		self.preprocessed_data_path = os.path.join(self.save_dir, 'preprocessed_data')
		self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.pkl")

		# load new study
		self.load_all_files = config.getboolean('load_new_study', 'load_all_files')
		if self.load_all_files:
			self.pattern_content = config.get('load_new_study', 'pattern_content', fallback='')
		self.extension = config['load_new_study']['extension']
		self.data_type = config['load_new_study']['data_type']
		self.filter_data = config.getboolean('load_new_study', 'filter_data')
		self.filter_method = config['load_new_study']['filter_method'] if self.filter_data else ''
		self.lowcut_freq = config.getint('load_new_study', 'lowcut_freq') if self.filter_data else ''
		self.highcut_freq = config.getint('load_new_study', 'highcut_freq') if self.filter_data else ''
		self.downsample_data = config.getboolean('load_new_study', 'downsample_data')
		self.sample_rate = config.getint('load_new_study', 'sample_rate') if self.downsample_data else ''
		self.remove_channels = config.getboolean('load_new_study', 'remove_channels')
		self.ch2rm = config['load_new_study']['ch2rm'] if self.remove_channels else 'missing'

		# clustering
		self.smoothing_gfp = config.getboolean('do_clustering', 'smoothing_gfp')
		self.smoothing_distance = config.getint('do_clustering', 'smoothing_distance') if self.smoothing_gfp else ''
		self.raw_features_path = os.path.join(self.save_dir, 'raw_features')
		self.hdf_concatenated_data_path = os.path.join(self.save_dir, self.study_name+'_concatenated_data.hdf')
		self.choose_number_of_maps = config['do_clustering']['choose_number_of_maps']
		self.number_of_maps = config.getint('do_clustering', 'number_of_maps')
		self.initializer = config['do_clustering']['initializer']
		self.clustering_method = config['do_clustering']['clustering_method']
		self.clustering_tolerance = config.getfloat('do_clustering', 'clustering_tolerance')
		need_options = ['X-means', 'Agglomerative hierarchical clustering', 'K-means']
		self.clustering_option = config['do_clustering']['clustering_option'] if self.clustering_method in need_options else ''
		self.number_of_repeats = config.getint('do_clustering', 'number_of_repeats')
		self.microstate_maps_path = os.path.join(self.raw_features_path, 'microstate_maps')

		# backfitting
		self.backfit_to = config['do_backfitting']['backfit_to']
		self.filter_segments = config.getboolean('do_backfitting', 'filter_segments')
		self.remove_segments_less_than = config.getint('do_backfitting', 'remove_segments_less_than') if self.filter_segments else ''
		self.filter_segments_option = config['do_backfitting']['filter_segments_option'] if self.filter_segments else ''
		self.micro_labels = ['A', 'B', 'C', 'D', 'E']

		# extract feature
		self.extracted_features_path = os.path.join(self.save_dir, 'extracted_features')
		self.hf_segmentation_path = os.path.join(self.raw_features_path, self.study_name+'_segmentation.hdf')
		self.raw_transitions_path = os.path.join(self.raw_features_path, 'raw_transitions')
		self.output_format = config['extract_features']['output_format']
		self.Features = [x for x in config['extract_features']['Features'].split(',')]
		self.save_transitions_bool = True if 'TP' in self.Features else False

		# source localize microstates
		self.localized_sources_path = os.path.join(self.raw_features_path, 'localized_sources')
		self.microstate_maps_path = os.path.join(self.raw_features_path, 'microstate_maps.csv')
		self.inverse_method = config['source_localize_microstates']['inverse_method']
		self.nperm = config.getint('source_localize_microstates', 'nperm')
		self.spacing = config['source_localize_microstates']['spacing']


	def load_new_study(self):
		if self.load_all_files:
			pattern = '*'
		else:
			pattern = '*'+self.pattern_content+'*'
		list_eegs = find_data(self.input_folder, self.extension, pattern)
		self.list_eegs = [os.path.basename(x).split('.')[0] for x in list_eegs]
		assert list_eegs, 'eeg list is empty'

		# Load channel location
		chan_loc_extension = os.path.basename(self.channel_location_dir).split('.')[-1]
		valid_chan_loc_extensions = ['loc', 'locs', 'eloc', 'sfp', 'csd', 'elc', 'txt',
										 'csd', 'elp', 'bvef', 'csv', 'tsv', 'xyz']
		assert chan_loc_extension in valid_chan_loc_extensions, ''' 
			Load Error, 
			File extension is expected to be: ‘.loc’ or ‘.locs’ or ‘.eloc’ (for EEGLAB files),
			‘.sfp’ (BESA/EGI files), ‘.csd’, ‘.elc’, ‘.txt’, ‘.csd’, ‘.elp’ (BESA spherical),
			‘.bvef’ (BrainVision files), ‘.csv’, ‘.tsv’, ‘.xyz’ (XYZ coordinates)
			'''

		if not os.path.exists(self.preprocessed_data_path):
			os.makedirs(self.preprocessed_data_path)


		print("preprocessing data ...")

		length_all_data = []
		counter = 1
		for filename in tqdm(list_eegs):
			progress, preprocessed_data, length_data, eeg_info, channels2remove = preprocess.preprocess_eegs(
					filename,
					list_eegs,
					self.extension,
					self.data_type,
					self.channel_location_dir,
					self.filter_data,
					self.filter_method,
					self.lowcut_freq,
					self.highcut_freq,
					self.downsample_data,
					self.sample_rate,
					self.ch2rm
					)
			# Save EEG info
			# TODO: save sample_rate to config file
			self.sample_rate = eeg_info['sfreq']
			self.preprocessed_data = preprocessed_data.T # TODO: test
			self.eeg_info = eeg_info

			save_eeg_info(self.eeg_info_path, eeg_info)

			ch_names, ch_location = list(eeg_info['ch_names']), eeg_info['chs']
			n_chan = len(ch_names)
			self.n_chan = n_chan
			self.ch_names = ch_names
			name = os.path.basename(filename)
			name = os.path.splitext(name)[0]
			save_path = os.path.join(self.preprocessed_data_path, name + ".hdf")
			with h5py.File(save_path, "w") as hf:
				dataset = hf.create_dataset(name, data=preprocessed_data, compression="gzip", compression_opts=9)
				# add metadata
				dataset.attrs['data_length'] = preprocessed_data.shape[1]
				dataset.attrs['eeg_format'] = self.extension
				dataset.attrs['data_type'] = self.data_type
				dataset.attrs['nchan'] = n_chan
				dataset.attrs['ch_names'] = ch_names
				dataset.attrs['ch_removed'] = self.ch2rm
				dataset.attrs['sample_rate'] = self.sample_rate
				dataset.attrs['filter_method'] = self.filter_method
				dataset.attrs['lowcut_freq'] = self.lowcut_freq
				dataset.attrs['highcut_freq'] = self.highcut_freq
			if counter == 1:
				filenames = filename
				catdata = preprocessed_data
			else:
				filenames = np.append(filenames, filename)
				catdata = np.append(catdata, preprocessed_data, axis=1)
			counter += 1
			
			self.length_all_data = np.append(length_all_data, int(length_data))
			# print("Progress:", progress, "%")

			# if progress == 100:
		print("\nSaving the concatenated data ...")
		# Save catdata
		catdata_filename = os.path.join(self.save_dir, self.study_name + "_concatenated_data.hdf")
		with h5py.File(catdata_filename, "w") as catf:
			catf.create_dataset(self.study_name, data=catdata, compression="gzip", compression_opts=9)

		# TODO: Write logs to config
		# Write "preprocessing settings" to config
		# Write "preprocessing results" to config
			

	def do_clustering(self):
		if not os.path.exists(self.raw_features_path):
			os.makedirs(self.raw_features_path)

		if self.smoothing_gfp:
			min_distance_size = int(self.smoothing_distance/(1000/self.sample_rate))
		else:
			min_distance_size = []

		avaliable_methods = ['Modified K-means',
							'Mini Batch Modified K-means',
							'K-means',
							'Mini Batch K-means',
							'X-means',
							'Agglomerative hierarchical clustering',
							'TTSAS',
							'BSAS',
							'CLARANS',
							'MBSAS',
							'OPTICS',
							'ROCK',
							'DBSCAN'
							]
		assert self.clustering_method in avaliable_methods, "clustering_method not supported"

		best_maps, gev, _ = clustering_func(
					self.preprocessed_data_path,
					self.hdf_concatenated_data_path,
					self.n_chan,
					self.clustering_method,
					self.number_of_maps,
					self.initializer,
					min_distance_size,
					self.number_of_repeats,
					self.clustering_tolerance,
					self.clustering_option
					)
		# microstate_maps = best_maps
		self.best_maps = best_maps

		# Save Maps
		maps_df = pd.DataFrame(best_maps.T, index=self.ch_names)
		maps_df.to_csv(self.microstate_maps_path + '.csv')

		# TODO: label maps

		print(f'Global Explained Variance: {gev}')
		# return best_maps


	def do_backfitting(self):
		print('\nBackfitting Maps to Data ...')
		backfit_func(self.study_name,
					 self.preprocessed_data_path,
					 self.best_maps,
					 self.backfit_to,
					 self.sample_rate,
					 self.filter_segments_option,
					 self.remove_segments_less_than,
					 self.micro_labels,
					 self.raw_features_path
					 )
	

	def extract_features_from_map(self):
		print("\nExtracting Features ...\n")

		if not os.path.exists(self.extracted_features_path):
			os.makedirs(self.extracted_features_path)

		length_data = [int(x) for x in self.length_all_data]
		extracted_features_df = extract_features(self.preprocessed_data_path,
														 self.hf_segmentation_path,
														 self.best_maps,
														 self.micro_labels,
														 self.sample_rate,
														 self.Features,
														 np.min(length_data)
														 )
		save_features(extracted_features_df, 'extracted_features',
							  self.output_format,
							  self.extracted_features_path)

		if self.save_transitions_bool:
			if not os.path.exists(self.raw_transitions_path):
				os.makedirs(self.raw_transitions_path)
			save_transitions(self.raw_features_path,
							 self.micro_labels,
							 self.output_format,
							 self.raw_transitions_path)
		print('\n*** Finished ***')

	def source_localize_microstates(self):
		print("Source Localizing Microstates ...")

		microstate_maps_df = pd.read_csv(self.microstate_maps_path)
		microstate_maps = np.asarray(microstate_maps_df.iloc[:, 1:])

		eeg_info = load_eeg_info(self.eeg_info_path)
		run_source_localization(self.preprocessed_data_path,
										self.localized_sources_path,
										eeg_info,
										microstate_maps,
										self.inverse_method,
										self.nperm,
										self.spacing)


def main():
	warnings.simplefilter("ignore")
	config_file = '/Users/bottlecap/Downloads/config.ini'
	config = load_config(config_file)
	
	study_name = config['base']['study_name']
	input_folder = config['base']['input_folder']
	channel_location_dir = config['base']['channel_location_dir']
	output_folder = config['base']['output_folder']

	print(study_name)
	print(input_folder)
	print(channel_location_dir)
	print(output_folder)
	

	tbx = ToolBox(config)
	# load a new study
	# just like press "New Study Button"
	
	# tbx.load_new_study()

	
	with open('/Users/bottlecap/Downloads/tbx.pkl', 'wb') as output:
		pickle.dump(tbx, output, pickle.HIGHEST_PROTOCOL)
	with open('/Users/bottlecap/Downloads/tbx.pkl', 'rb') as input_tbx:
		tbx = pickle.load(input_tbx)

	# for i in range(0, tbx.preprocessed_data.shape[0], 200):
	# 	fig, axes = plt.subplots(1)  # assuming 3 channel types
	# 	mne.viz.plot_topomap(tbx.preprocessed_data[i, :], tbx.eeg_info, axes=axes, sensors=False, show=False)
	# 	fig.savefig(f'/Users/bottlecap/Downloads/eeg_images/res{i}.png')
	# 	if i > 2000:
	# 		break
	# do clustering
	# just like press "start clustering"
	tbx.do_clustering()

	# do backfitting
	# just like press "Start Backfitting"
	tbx.do_backfitting()

	tbx.extract_features_from_map()
	tbx.source_localize_microstates()


if __name__ == '__main__':
	main()




