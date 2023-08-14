from functions.utils.data_io import load_config
import warnings
from ToolBox import ToolBox
import pickle
import matplotlib.pyplot as plt
import mne
# from functions.utils.extract_peaks_maps import extract_peaks_maps

def main():
	warnings.simplefilter("ignore")
	config_file = '/Users/bottlecap/EEG-Microstate-Feature-Extraction/src/main/python/microstate_tool/config.ini'
	config = load_config(config_file)
	
	study_name = config['base']['study_name']
	input_folder = config['base']['input_folder']
	channel_location_dir = config['base']['channel_location_dir']
	output_folder = config['base']['output_folder']

	print(study_name)
	print(input_folder)
	print(channel_location_dir)
	print(output_folder)

	new_tbx = True
	if new_tbx:
		tbx = ToolBox(config)
	else:
		with open('/Users/bottlecap/Downloads/output/xxx/tbx_object.pkl', 'rb') as input_tbx:
			tbx = pickle.load(input_tbx)
	process = [
		tbx.load_raw,
		tbx.load_channel_location,
		tbx.load_new_study,
		tbx.do_clustering,
		tbx.do_labeling,
		tbx.do_backfitting,
		tbx.extract_features_from_map,
		# tbx.source_localize_microstates
	]

	for i in process:
		i()



if __name__ == '__main__':
	main()








		# hdf_concatenated_data_path = '/Users/bottlecap/Downloads/output/test_tbx_4/test_tbx_4_concatenated_data.hdf'
	# concatenated_data = import_hdf_data(hdf_concatenated_data_path)
	# min_distance_size = int(tbx.smoothing_distance/(1000/tbx.sample_rate))
	# all_maps, peaks = extract_peaks_maps(concatenated_data, min_distance_size)
	# print(all_maps.shape)
	# kmeans = pickle.load(open('/Users/bottlecap/Downloads/kmeans.pkl', 'rb'))
	# for i in range(0, all_maps.shape[0], 10):
	# 	fig, axes = plt.subplots(1)  # assuming 3 channel types
	# 	mne.viz.plot_topomap(all_maps[i, :], tbx.eeg_info, axes=axes, sensors=False, show=False)
	# 	result = kmeans.predict(all_maps[i, :].reshape((1, -1)))
	# 	# print(all_maps[i, :].shape)
	# 	fig.savefig(f'/Users/bottlecap/Downloads/eeg_images/res_{result[0]}_{i}.png')
	# 	if i > 3000:
	# 		break