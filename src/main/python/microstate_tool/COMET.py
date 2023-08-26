
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from functions.sourcelocalization_utils.source_localization_functions import run_source_localization #, visualize_sources
import pickle

from functions.data_utils.data_io import DataIO
from functions.data_utils.data_preprocessor import DataPreprocessor
from functions.features_utils.feature_extractor import FeatureExtractor
from functions.features_utils.feature_io import FeatureIO
from functions.backfitting_utils.segmentation_io import SegmentationIO
from functions.backfitting_utils.microstate_backfitter import MicrostateBackfitter
from functions.clustering_utils.microstate_clusterer import clustering_func


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
		# self.done_extracting_microsegments = False
		self.done_source_localization = False
		self.auto_save = auto_save

	def load_config(self, config):
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
		self.datatype = config['load_new_study']['datatype']
		self.filter_data = config.getboolean('load_new_study', 'filter_data')
		self.filter_method = config['load_new_study']['filter_method'] if self.filter_data else ''
		self.lowcut_freq = config.getint('load_new_study', 'lowcut_freq') if self.filter_data else ''
		self.highcut_freq = config.getint('load_new_study', 'highcut_freq') if self.filter_data else ''
		self.downsample_data = config.getboolean('load_new_study', 'downsample_data')
		self.sample_rate = config.getint('load_new_study', 'sample_rate') if self.downsample_data else ''
		self.remove_channels = config.getboolean('load_new_study', 'remove_channels')
		self.ch2rm = config['load_new_study']['ch2rm'] if self.remove_channels else 'missing'

		# clustering_utils
		self.smoothing_gfp = config.getboolean('do_clustering', 'smoothing_gfp')
		self.smoothing_distance = config.getint('do_clustering', 'smoothing_distance') if self.smoothing_gfp else ''
		self.raw_features_path = os.path.join(self.save_dir, 'raw_features')
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
		need_options = ['X-means', 'Agglomerative hierarchical clustering_utils', 'K-means']
		self.clustering_option = config['do_clustering']['clustering_option'] if self.clustering_method in need_options else ''
		self.number_of_repeats = config.getint('do_clustering', 'number_of_repeats')
		self.microstate_maps_path = os.path.join(self.raw_features_path, 'microstate_maps.csv')
		self.use_percentages = config.getint('do_clustering', 'use_percentages')

		# backfitting_utils
		self.backfit_to = config['do_backfitting']['backfit_to']
		self.filter_segments = config.getboolean('do_backfitting', 'filter_segments')
		self.remove_segments_less_than = config.getint('do_backfitting', 'remove_segments_less_than') if self.filter_segments else ''
		self.filter_segments_option = config['do_backfitting']['filter_segments_option'] if self.filter_segments else ''
		self.epsilon = config.getfloat('do_backfitting', 'epsilon')
		self.b = config.getint('do_backfitting', 'b')
		self.lamb = config.getint('do_backfitting', 'lamb')
		# TODO: use models to automatically label them
		self.micro_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

		# extract feature
		self.extracted_features_path = os.path.join(self.save_dir, 'extracted_features')
		self.segmentation_path = os.path.join(self.save_dir, 'segmentations')
		self.raw_transitions_path = os.path.join(self.raw_features_path, 'raw_transitions')
		self.export_format = config['extract_features']['export_format']
		self.feature_list = [x for x in config['extract_features']['feature_list'].split(',')]
		self.save_transitions_bool = True if 'TP' in self.feature_list else False
		self.window_size = config.getint('extract_features', 'window_size') if 'OCC' in self.feature_list else ''

		# source localize microstates
		self.localized_sources_path = os.path.join(self.raw_features_path, 'localized_sources')
		# self.microstate_maps_path = os.path.join(self.raw_features_path, 'microstate_maps.csv')
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

	def load_new_study(self):
		
		if not os.path.exists(self.preprocessed_data_path):
			os.makedirs(self.preprocessed_data_path)

		length_all_data = []
		data_io = DataIO()
		counter = 1
		for filename in tqdm(self.list_eegs_path):
			# Create an instance of the DataPreprocessor class
			preprocessor = DataPreprocessor()
			progress, eeg, preprocessed_data, length_data, eeg_info, channels2remove = preprocessor.preprocess_eegs(
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
			# print("Progress:", progress, "%")

			# save EEG object
			print(f"\nPreprocessing file: {filename}")
			data_io.export_eegs(eeg, save_path, self.extension, self.datatype)

		self.done_preprocessing = True
		if self.auto_save:
			self.save_tbx()
			

	def do_clustering(self):
		if not os.path.exists(self.raw_features_path):
			os.makedirs(self.raw_features_path)

		if self.smoothing_gfp:
			self.min_distance_size = int(self.smoothing_distance/(1000/self.sample_rate))
		else:
			self.min_distance_size = []

		avaliable_methods = ['Modified K-means',
							'K-means',
							'X-means',
							'Agglomerative hierarchical clustering_utils',
							]
		assert self.clustering_method in avaliable_methods, "clustering_method not supported"

		best_maps, gev, _, n_states = clustering_func(
					self.preprocessed_data_path,
					self.extension,
					self.datatype,
					self.n_chan,
					self.clustering_method,
					self.number_of_maps,
					self.initializer,
					self.use_percentages,
					self.min_distance_size,
					self.max_iterations,
					self.clustering_tolerance,
					self.number_of_repeats,
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
		# TODO: label maps
		self.micro_labels = self.micro_labels[:self.n_states]
		self.done_labeling_microstates = True


	def do_backfitting(self):
		if not os.path.exists(self.segmentation_path):
			os.makedirs(self.segmentation_path)
		print('\nBackfitting Maps to Data ...')

		backfitter_instance = MicrostateBackfitter(
			self.study_name,
			self.preprocessed_data_path,
			self.best_maps,
			self.backfit_to,
			self.filter_segments_option,
			self.remove_segments_less_than,
			self.micro_labels,
			self.segmentation_path,
			self.extension,
			self.datatype,
			[self.epsilon, self.b, self.lamb],
			self.export_format)
		backfitter_instance.perform_segmentation()

		self.done_backfitting = True
		if self.auto_save:
			self.save_tbx()


	def extract_features(self):
		if not os.path.exists(self.extracted_features_path):
			os.makedirs(self.extracted_features_path)

		print("\nExtracting Static Features ...\n")

		segmentation_list_path, segmentation_list_filename = DataIO().find_data(
			self.segmentation_path,
			self.export_format
		)

		for s in range(len(segmentation_list_path)):

			segmentation_path = segmentation_list_path[s]
			filename = segmentation_list_filename[s]
			print(filename)

			segmentation_array = SegmentationIO().load_segmentation(segmentation_path, import_format='.csv')

			#self.feature_mode = ['static', 'dynamic']
			if 'static' in self.feature_mode:
				static_features_path = os.path.join(self.extracted_features_path, 'static')
				if not os.path.exists(static_features_path):
					os.makedirs(static_features_path)
				feature_extractor = FeatureExtractor(segmentation_array,
													 self.sample_rate,
													 self.window_size,
													 mode='static')
				feature_io = FeatureIO()

				# TODO: export features
				print(self.feature_list)
				if 'COV' in self.feature_list:
					static_microstate_coverage = feature_extractor.microstate_coverage()
					print(static_microstate_coverage)
					feature_io.export_features(static_microstate_coverage, 'static', static_features_path, filename,
											   self.export_format)
				if 'OCC' in self.feature_list:
					static_microstate_occurrence = feature_extractor.microstate_occurrence()
					print(static_microstate_occurrence)
					feature_io.export_features(static_microstate_occurrence, 'static', static_features_path, filename,
											   self.export_format)
				if 'DUR' in self.feature_list:
					static_microstate_duration = feature_extractor.microstate_duration()
					print(static_microstate_duration)
					feature_io.export_features(static_microstate_duration, 'static', static_features_path, filename,
											   self.export_format)
				if 'TP' in self.feature_list:
					static_microstate_transition_probability = feature_extractor.transition_probability()
					feature_io.export_features(static_microstate_transition_probability, 'static', static_features_path,
											   filename,
											   self.export_format)
				if 'LZC' in self.feature_list:
					static_microstate_complexity = feature_extractor.lempel_ziv_complexity()
					feature_io.export_features(static_microstate_complexity, 'static', static_features_path, filename,
											   self.export_format)

			if 'dynamic' in self.feature_mode:
				dynamic_features_path = os.path.join(self.extracted_features_path, 'dynamic')
				if not os.path.exists(dynamic_features_path):
					os.makedirs(dynamic_features_path)

				feature_extractor = FeatureExtractor(segmentation_array,
													 self.sample_rate,
													 self.window_size,
													 mode='dynamic')
				feature_io = FeatureIO()
				if 'COV' in self.feature_list:
					dynamic_microstate_coverage = feature_extractor.microstate_coverage()
					feature_io.export_features(dynamic_microstate_coverage, 'dynamic', dynamic_features_path, filename,
											   self.export_format)
				if 'OCC' in self.feature_list:
					dynamic_microstate_occurrence = feature_extractor.microstate_occurrence()
					feature_io.export_features(dynamic_microstate_occurrence, 'dynamic', dynamic_features_path,
											   filename,
											   self.export_format)
				if 'DUR' in self.feature_list:
					dynamic_microstate_duration = feature_extractor.microstate_duration()
					feature_io.export_features(dynamic_microstate_duration, 'dynamic', dynamic_features_path, filename,
											   self.export_format)
				if 'TP' in self.feature_list:
					dynamic_microstate_transition_probability = feature_extractor.transition_probability()
					feature_io.export_features(dynamic_microstate_transition_probability, 'dynamic',
											   dynamic_features_path,
											   filename,
											   self.export_format)
				if 'LZC' in self.feature_list:
					dynamic_microstate_complexity = feature_extractor.lempel_ziv_complexity()
					feature_io.export_features(dynamic_microstate_complexity, 'dynamic', dynamic_features_path,
											   filename,
											   self.export_format)


		"""
		if self.save_transitions_bool:
			if not os.path.exists(self.raw_transitions_path):
				os.makedirs(self.raw_transitions_path)
			save_transitions(self.raw_features_path,
							 self.micro_labels,
							 self.export_format,
							 self.raw_transitions_path)
	 	"""
		print('\n*** Finished ***')
		self.done_extracting_features = True
		if self.auto_save:
			self.save_tbx()

	def source_localize_microstates(self):
		print("Source Localizing Microstates ...")

		microstate_maps_df = pd.read_csv(self.microstate_maps_path)
		microstate_maps = np.asarray(microstate_maps_df.iloc[:, 1:])
		self.subjects_dir = 'fsaverage'

		run_source_localization(self.preprocessed_data_path,
										self.segmentation_path,
										self.localized_sources_path,
										self.subjects_dir, #'fsaverage', # TODO:
										microstate_maps,
										self.inverse_method,
										self.nperm,
										self.spacing,
										self.source_localization_method,
										self.extension,
										self.datatype)
		self.done_source_localization = True
		if self.auto_save:
			self.save_tbx()

	def save_tbx(self):
		self.tbx_object_path = os.path.join(self.save_dir, 'tbx_object.pkl')
		with open(self.tbx_object_path, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)






