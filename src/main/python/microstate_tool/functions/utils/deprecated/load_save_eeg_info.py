
import pickle


def save_eeg_info(eeg_info_path, eeg_info):
    with open(eeg_info_path, 'wb') as f:
        pickle.dump(eeg_info, f)


def load_eeg_info(eeg_info_path):
    with open(eeg_info_path, 'rb') as f:
        eeg_info = pickle.load(f)
    return eeg_info
