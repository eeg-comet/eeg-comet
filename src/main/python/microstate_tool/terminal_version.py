from functions.gui_utils.config_io import load_config
import warnings
from COMET import COMET
import pickle
import os


def main():
	warnings.simplefilter("ignore")
	# ==================================================================
	# TODO#1 set config file path here
	config_file = './config.ini'
	# ==================================================================
	config = load_config(config_file)
	
	study_name = config['base']['study_name']
	input_folder = config['base']['input_folder']
	channel_location_dir = config['base']['channel_location_dir']
	output_folder = config['base']['output_folder']

	# For config file double checking
	print(study_name)
	print(input_folder)
	print(channel_location_dir)
	print(output_folder)

	new_tbx = True
	if new_tbx:
		tbx = COMET(config)
	else:
		save_dir = os.path.join(output_folder, study_name)
		tbx_path = os.path.join(save_dir, 'comet_tbx_object.pkl')
		with open(tbx_path, 'rb') as input_tbx:
			tbx = pickle.load(input_tbx)

	# ==================================================================
	# # TODO#2 choose funtion for the toolbox (to run)
	process = [
		tbx.load_raw,
		tbx.load_channel_location,
		tbx.load_new_study,
		tbx.do_clustering,
		tbx.do_labeling,
		# tbx.do_backfitting,
		# tbx.extract_features_from_map,
		# tbx.source_localize_microstates
	]
	# ==================================================================

	for i in process:
		i()



if __name__ == '__main__':
	main()







