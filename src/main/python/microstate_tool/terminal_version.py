from functions.utils.data_io import load_config
import warnings
from ToolBox import ToolBox
import pickle

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
	tbx.load_raw()
	tbx.load_channel_location()
	tbx.load_new_study()
	tbx.save_tbx()

	
	# with open('/Users/bottlecap/Downloads/tbx.pkl', 'wb') as output:
	# 	pickle.dump(tbx, output, pickle.HIGHEST_PROTOCOL)
	# with open('/Users/bottlecap/Downloads/output/xxx/tbx_object.pkl', 'rb') as input_tbx:
	# 	tbx = pickle.load(input_tbx)

	# for i in range(0, tbx.preprocessed_data.shape[0], 200):
	# 	fig, axes = plt.subplots(1)  # assuming 3 channel types
	# 	mne.viz.plot_topomap(tbx.preprocessed_data[i, :], tbx.eeg_info, axes=axes, sensors=False, show=False)
	# 	fig.savefig(f'/Users/bottlecap/Downloads/eeg_images/res{i}.png')
	# 	if i > 2000:
	# 		break
	# do clustering
	# just like press "start clustering"
	tbx.do_clustering()
	tbx.do_labeling()
	tbx.save_tbx()
	# do backfitting
	# just like press "Start Backfitting"
	tbx.do_backfitting()
	tbx.save_tbx()

	tbx.extract_features_from_map()
	tbx.save_tbx()
	# tbx.source_localize_microstates()
	# tbx.save_tbx()


if __name__ == '__main__':
	main()