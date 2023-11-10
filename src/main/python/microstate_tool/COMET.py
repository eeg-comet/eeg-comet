
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import mne
import cv2
import io
from keras.models import load_model
import matplotlib.pyplot as plt

from functions.data_utils.data_io import DataIO
from functions.data_utils.data_preprocessor import DataPreprocessor
from functions.features_utils.feature_extractor import FeatureExtractor
from functions.features_utils.feature_io import FeatureIO
from functions.backfitting_utils.segmentation_io import SegmentationIO
from functions.backfitting_utils.microstate_backfitter import MicrostateBackfitter
from functions.clustering_utils.microstate_clusterer import MicrostateClusterer
from functions.sourcelocalization_utils.source_localizer import SourceLocalizer

class COMET:
	def __init__(self, config=None, auto_save=True):
		'''
		config: should be a ConfigParser, use it to load configs
		auto_save: if True, the COMET will automatically save itself after each process
		'''
		if config:
			self.load_config(config)
		self.done_preprocessing = False
		self.done_clustering = False
		self.done_labeling_microstates = False
		self.done_backfitting = False
		self.done_extracting_features = False
		self.done_source_localization = False
		self.done_source_microstate_correlation = False
		self.auto_save = auto_save
		self.log_text = ""

	def load_config(self, config):
		# Base configs
		self.config = config
		self.study_name = config['base']['study_name']
		self.input_folder = config['base']['input_folder']
		self.channel_location_dir = config['base']['channel_location_dir']
		self.output_folder = config['base']['output_folder']

		self.save_dir = os.path.join(self.output_folder, self.study_name)
		assert self.study_name != "", "study name cannot be empty"
		self.preprocessed_data_path = os.path.join(self.save_dir, self.study_name+'_preprocessed_data')
		self.eeg_info_path = os.path.join(self.save_dir, "eeg_info.pkl")

		# Load new study configs
		self.load_all_files = config.getboolean('load_new_study', 'load_all_files')
		if self.load_all_files:
			self.pattern_content = config.get('load_new_study', 'pattern_content', fallback='')
		self.extension = config['load_new_study']['extension']
		self.datatype = config['load_new_study']['datatype']
		self.filter_data = config.getboolean('load_new_study', 'filter_data')
		self.filter_method = config['load_new_study']['filter_method'] if self.filter_data else ''
		self.lowcut_freq = config.getint('load_new_study', 'lowcut_freq') if self.filter_data else ''
		self.highcut_freq = config.getint('load_new_study', 'highcut_freq') if self.filter_data else ''
		self.downsample_data = config.getboolean('load_new_study', 'downsample_data')
		self.sample_rate = config.getint('load_new_study', 'sample_rate') if self.downsample_data else ''
		self.remove_channels = config.getboolean('load_new_study', 'remove_channels')
		self.ch2rm = config['load_new_study']['ch2rm'] if self.remove_channels else 'missing'

		# Clustering configs
		self.smoothing_gfp = config.getboolean('do_clustering', 'smoothing_gfp')
		self.smoothing_distance = config.getint('do_clustering', 'smoothing_distance') if self.smoothing_gfp else ''
		number_of_maps = config['do_clustering']['number_of_maps']
		self.number_of_maps = number_of_maps if number_of_maps=='auto' else int(number_of_maps)
		self.choose_number_of_maps = 'Auto' if number_of_maps == 'auto' else 'User'
		self.stopping_mode = config['do_clustering']['stopping_mode'] if self.number_of_maps == 'auto' else ''
		self.stopping_parameter = config.getfloat('do_clustering', 'stopping_parameter') if self.number_of_maps == 'auto' else ''
		self.kmin = config.getint('do_clustering', 'kmin') if self.number_of_maps == 'auto' else ''
		self.kmax = config.getint('do_clustering', 'kmax') if self.number_of_maps == 'auto' else ''
		self.initializer = config['do_clustering']['initializer']
		self.clustering_method = config['do_clustering']['clustering_method']
		self.max_iterations = config.getint('do_clustering', 'max_iterations')
		self.clustering_tolerance = config.getfloat('do_clustering', 'clustering_tolerance')
		need_options = ['X-Means Clustering', 'Agglomerative Hierarchical Clustering', 'K-Means Clustering', 'PCA + K-Means Clustering',
							 'Autoencoder + K-Means Clustering',]
		self.clustering_option = config['do_clustering']['clustering_option'] if self.clustering_method in need_options else ''
		self.number_of_repeats = config.getint('do_clustering', 'number_of_repeats')
		self.microstate_maps_path = os.path.join(self.save_dir, 'microstate_maps.csv')
		self.use_percentages = config.getint('do_clustering', 'use_percentages')

		# Backfitting configs
		self.backfit_to = config['do_backfitting']['backfit_to']
		self.identify_short_window = config.getboolean('do_backfitting', 'identify_short_window')
		self.filter_segments = config.getboolean('do_backfitting', 'filter_segments')
		self.remove_segments_less_than = config.getint('do_backfitting', 'remove_segments_less_than') if self.filter_segments else ''
		self.filter_segments_option = config['do_backfitting']['filter_segments_option'] if self.filter_segments else ''
		self.epsilon = config.getfloat('do_backfitting', 'epsilon')
		self.b = config.getint('do_backfitting', 'b')
		self.lamb = config.getint('do_backfitting', 'lamb')
		# TODO: use models to automatically label them
		self.micro_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

		# Feature extraction configs
		self.extracted_features_path = os.path.join(self.save_dir, 'extracted_features')
		self.segmentation_path = os.path.join(self.save_dir, 'segmentations')
		self.export_format = config['extract_features']['export_format']
		self.feature_list = [x for x in config['extract_features']['feature_list'].split(',')]
		self.save_transitions_bool = True if 'TP' in self.feature_list else False
		self.window_size = config.getint('extract_features', 'window_size') if 'OCC' in self.feature_list else ''

		# Source localization configs
		self.localized_sources_path = os.path.join(self.save_dir, 'localized_sources')
		self.inverse_method = config['source_localize_microstates']['inverse_method']
		self.nperm = config.getint('source_localize_microstates', 'nperm')
		self.spacing = config['source_localize_microstates']['spacing']
		self.source_localization_method = config['source_localize_microstates']['source_localization_method']

	def load_raw(self):
		'''
		get all avaliable eeg file path
		'''
		if self.load_all_files:
			self.pattern = '*'
		else:
			self.pattern = '*'+self.pattern_content+'*'

		self.list_eegs_path, self.list_eegs = DataIO().find_data(self.input_folder, self.extension, self.pattern)
		#self.list_eegs = [os.path.basename(x).split('.')[0] for x in self.list_eegs_path]
		#assert self.list_eegs_path, 'eeg list is empty'

	def load_channel_location(self):
		# Load channel location
		if os.path.isfile(self.channel_location_dir):
			chan_loc_extension = os.path.basename(self.channel_location_dir).split('.')[-1]
			valid_chan_loc_extensions = ['loc', 'locs', 'eloc', 'sfp', 'csd', 'elc', 'txt',
											 'csd', 'elp', 'bvef', 'csv', 'tsv', 'xyz']
			assert chan_loc_extension in valid_chan_loc_extensions, ''' 
				Load Error, 
				File extension is expected to be: ‘.loc’ or ‘.locs’ or ‘.eloc’ (for EEGLAB files),
				‘.sfp’ (BESA/EGI files), ‘.csd’, ‘.elc’, ‘.txt’, ‘.csd’, ‘.elp’ (BESA spherical),
				‘.bvef’ (BrainVision files), ‘.csv’, ‘.tsv’, ‘.xyz’ (XYZ coordinates)
				'''
		#return chan_loc_extension in valid_chan_loc_extensions

	def do_preprocessing(self):

		if not os.path.exists(self.preprocessed_data_path):
			os.makedirs(self.preprocessed_data_path)

		length_all_data = []
		data_io = DataIO()

		progress_bar = tqdm(total=len(self.list_eegs), ncols=100, position=0, leave=True)

		for filename in self.list_eegs_path:
			progress_bar.set_description(f"Preprocessing file: {self.list_eegs[self.list_eegs_path.index(filename)]}")
			# Create an instance of the DataPreprocessor class
			preprocessor = DataPreprocessor()
			eeg, preprocessed_data, length_data, eeg_info, channels2remove = preprocessor.preprocess_eegs(
					filename,
					self.list_eegs_path,
					self.extension,
					self.datatype,
					self.channel_location_dir,
					self.filter_data,
					self.filter_method,
					self.lowcut_freq,
					self.highcut_freq,
					self.downsample_data,
					self.sample_rate,
					self.ch2rm
					)
			progress_bar.update(1)

			self.eeg_info = eeg.info
			# Save EEG info
			if not self.sample_rate:
				self.sample_rate = int(self.eeg_info['sfreq'])
			self.channels2remove = channels2remove

			data_io.save_eeg_info(self.eeg_info_path, eeg_info)

			ch_names, ch_location = list(self.eeg_info['ch_names']), self.eeg_info['chs']
			n_chan = len(ch_names)
			self.n_chan = n_chan
			self.ch_names = ch_names
			name = os.path.basename(filename)
			name = os.path.splitext(name)[0]
			save_path = os.path.join(self.preprocessed_data_path, name)
			
			self.length_all_data = np.append(length_all_data, int(length_data))

			# save EEG object
			data_io.export_eegs(eeg, save_path, self.extension, self.datatype)

		progress_bar.close()

		self.done_preprocessing = True
		if self.auto_save:
			self.save_tbx()
			

	def do_clustering(self):
		print('\nClustering ...')

		if self.smoothing_gfp:
			self.min_distance_size = int(self.smoothing_distance/(1000/self.sample_rate))
		else:
			self.min_distance_size = []

		avaliable_methods = ['Modified K-Means Clustering',
							 'K-Means Clustering',
							 'PCA + K-Means Clustering',
							 'Autoencoder + K-Means Clustering',
							 'X-Means Clustering',
							 'Agglomerative Hierarchical Clustering',
							 ]
		assert self.clustering_method in avaliable_methods, "clustering_method not supported"
		microstate_clusterer = MicrostateClusterer(self.number_of_repeats, self.max_iterations, self.clustering_tolerance)
		best_maps, gev, _, n_states = microstate_clusterer.clustering_func(
					self.preprocessed_data_path,
					self.extension,
					self.datatype,
					self.n_chan,
					self.clustering_method,
					self.number_of_maps,
					self.initializer,
					self.use_percentages,
					self.min_distance_size,
					self.clustering_option,
					self.stopping_mode,
					self.stopping_parameter,
					self.kmin,
					self.kmax
					)
		# microstate_maps = best_maps
		self.best_maps = np.array(best_maps)
		self.n_states = n_states

		# Save Maps
		maps_df = pd.DataFrame(self.best_maps.T, index=self.ch_names)
		maps_df.to_csv(self.microstate_maps_path)

		
		self.gev = gev
		print(f'Global Explained Variance: {self.gev}')
		self.done_clustering = True
		# return best_maps
		if self.auto_save:
			self.save_tbx()

	def do_labeling(self):
		image_size = 448
		images = []
		for i in range(self.best_maps.shape[0]):
			fig, ax = plt.subplots()
			mne.viz.plot_topomap(self.best_maps[i, :], self.eeg_info, contours=10, sensors=False, axes=ax, show=False, sphere='auto')
			with io.BytesIO() as buf:
				fig.savefig(buf, dpi=200, bbox_inches='tight')
				buf.seek(0)
				img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
			image = cv2.imdecode(img_arr, 1)
			image = cv2.resize(image, (image_size, image_size))
			# cv2.imshow('1', image)
			# cv2.waitKey(0)
			image = np.expand_dims(image, axis=0)
			images.append(image)
		image = np.vstack(images)
		print(image.shape)

		# load model and do inference
		maps = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}
		# model_path = 
		directory = os.getcwd()
		directory = os.path.basename(directory)
		print(directory)
		# TODO: need to find a better way to distinguish the path
		# currently just for start the toolbox app from different path(GUI and terminal version)
		if directory == 'EEG-Microstate-Feature-Extraction':
			model_path = './src/main/python/microstate_tool/models/model_v1.11.h5' 
		else:
			model_path = './models/model_v1.11.h5'
		model = load_model(model_path, compile = False)
		output = model.predict(image)
		label_result = self.get_labels(output, maps)

		# show image for debugging
		# for i in images:
		# 	cv2.imshow(f'{i}', i.squeeze())
		# 	cv2.waitKey(0)

		micro_labels = []
		additional_label = 'M'
		assert self.n_states < 27, 'cannot label microstates more than 27: number of letters is not enough'
		for i in range(self.n_states):
			if i in label_result:
				micro_labels.append(label_result[i])
			else:
				micro_labels.append(additional_label)
				additional_label = chr(ord(additional_label)+1)
			# for other maps(>7), manually label them(leave them empty)
		
		# self.micro_labels = self.micro_labels[:self.n_states]
		self.micro_labels = micro_labels
		print(self.micro_labels)
		self.done_labeling_microstates = True


	def get_labels(self, confidences, maps):
		image_pool = set()
		state_pool = set()
		result = {}
		print(confidences)
		# print(confidences.argmax())
		# confidences[0, 1] = 8.2
		while True:
			# get max value's coordinate
			image = confidences.argmax() // confidences.shape[1]
			state = confidences.argmax() % confidences.shape[1]
			# print(image, state)
			if image not in image_pool and state not in state_pool:
				image_pool.add(image)
				state_pool.add(state)
				confidences[image, state] = np.NINF
				result[image] = maps[state]
			else:
				confidences[image, state] = np.NINF
			if len(result) == confidences.shape[0] or len(result) == 7:
				break
		print(result)
		return result


	def do_backfitting(self):
		print('\nBackfitting ...')

		if not os.path.exists(self.segmentation_path):
			os.makedirs(self.segmentation_path)

		backfitter_instance = MicrostateBackfitter(
			self.study_name,
			self.preprocessed_data_path,
			self.best_maps,
			self.backfit_to,
			self.filter_segments,
			self.filter_segments_option,
			self.identify_short_window,
			self.remove_segments_less_than,
			self.micro_labels,
			self.segmentation_path,
			self.extension,
			self.datatype,
			self.sample_rate,
			[self.epsilon, self.b, self.lamb],
			self.export_format,
		)
		backfitter_instance.perform_segmentation()

		self.done_backfitting = True
		if self.auto_save:
			self.save_tbx()


	def extract_features(self):
		print('\nExtracting Features ...')

		if not os.path.exists(self.extracted_features_path):
			os.makedirs(self.extracted_features_path)

		segmentation_list_path, segmentation_list_filename = DataIO().find_data(
			self.segmentation_path,
			self.export_format
		)

		# Create a tqdm progress bar
		progress_bar = tqdm(total=len(segmentation_list_path), ncols=100, position=0, leave=True)

		for s in range(len(segmentation_list_path)):

			segmentation_path = segmentation_list_path[s]
			filename = segmentation_list_filename[s]
			progress_bar.set_description(f"Extracting features: {filename}")

			segmentation_array = SegmentationIO().load_segmentation(segmentation_path, import_format='.csv')

			if 'static' in self.feature_mode:
				feature_extractor = FeatureExtractor(segmentation_array,
													 self.sample_rate,
													 self.window_size,
													 mode='static')
				if 'GEV' in self.feature_list:
					data_io = DataIO()
					eeg_path = os.path.join(self.preprocessed_data_path, f"{filename}{self.extension}")
					eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
					eeg_data = data_io.get_eeg_data(eeg, self.datatype)
					output_features = feature_extractor.extract_microstate_features(filename, self.feature_list, eeg_data,
																		 self.best_maps, self.micro_labels)
				else:
					output_features = feature_extractor.extract_microstate_features(filename, self.feature_list)
				if s == 0:
					static_features_dfs = output_features
				else:
					static_features_dfs = pd.concat([static_features_dfs, output_features], ignore_index=True)

			if 'dynamic' in self.feature_mode:
				feature_extractor = FeatureExtractor(segmentation_array,
													 self.sample_rate,
													 self.window_size,
													 mode='dynamic')

				if 'GEV' in self.feature_list:
					data_io = DataIO()
					eeg_path = os.path.join(self.preprocessed_data_path, f"{filename}{self.extension}")
					eeg = data_io.load_eegs(eeg_path, self.extension, self.datatype)
					eeg_data = data_io.get_eeg_data(eeg, self.datatype)
					output_features = feature_extractor.extract_microstate_features(filename, self.feature_list, eeg_data,
																		 self.best_maps, self.micro_labels)
				else:
					output_features = feature_extractor.extract_microstate_features(filename, self.feature_list)
				if s == 0:
					dynamic_features_dfs = output_features
				else:
					dynamic_features_dfs = pd.concat([dynamic_features_dfs, output_features], ignore_index=True)

			progress_bar.update(1)

		progress_bar.close()

		feature_io = FeatureIO()
		if 'static' in self.feature_mode:
			feature_io.export_features(
				static_features_dfs,
				'static',
				self.extracted_features_path,
				self.export_format
			)
		if 'dynamic' in self.feature_mode:
			feature_io.export_features(
				dynamic_features_dfs,
				'dynamic',
				self.extracted_features_path,
				self.export_format
			)

		self.done_extracting_features = True
		if self.auto_save:
			self.save_tbx()

	def source_localize_microstates(self):
		print("Source Localizing Microstates ...")

		microstate_maps_df = pd.read_csv(self.microstate_maps_path)
		microstate_maps = np.asarray(microstate_maps_df.iloc[:, 1:])


		if self.use_anatomy == "fsaverage":
			fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
			subjects_dir = os.path.dirname(fs_dir)
		elif self.use_anatomy == "individual":
			subjects_dir = self.individual_subjects_dir

		source_localizer = SourceLocalizer(subjects_dir,
										   self.localized_sources_path,
										   self.preprocessed_data_path,
										   self.segmentation_path,
										   self.use_anatomy,
										   self.extension,
										   self.datatype,
										   self.spacing,
										   self.inverse_method,
										   self.best_maps,
										   self.nperm)
		source_localizer.run_source_localization()

		self.done_source_localization = True
		if self.auto_save:
			self.save_tbx()

	def source_microstate_correlation(self):
		print("Correlating sources and microstates ...")

	# 	TODO: Need to expand this function to include the following:
	# 	1. Load the source localized time series
	# 	2. Load the microstates
	# 	3. Run TESS and Averaging based on the user input
	# 	4. Save the results

	def save_tbx(self):
		self.tbx_object_path = os.path.join(self.save_dir, 'comet_tbx_object.pkl')
		with open(self.tbx_object_path, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)






