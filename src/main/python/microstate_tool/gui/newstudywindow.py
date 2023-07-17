import os.path
import re
import numpy as np
import h5py
from PyQt5 import uic
import shutil
import mne
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox

from functions.utils.set_widgets_status import set_widgets_status
from functions.utils.data_io import find_data, load_eegs, save_eeg_info, initialize_config, load_config, save_config#, export_h5
from functions import preprocess

class NewStudyWindow(QDialog):

    def __init__(self, context, parent=None, main_window=None):
        super(NewStudyWindow, self).__init__(parent)
        
        # if main_window:
        self.main_window = main_window
        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("NewStudyWindow.ui"), self)
        
        self.ui.setWindowTitle("New Study - Import Raw Data and Preprocess")
        self.done_preprocessing = False
        self.channel_location_dir = ''

        self.ui.step0_load_all_radio.clicked.connect(self.newstudy_controller)
        self.ui.step0_load_pattern_radio.clicked.connect(self.newstudy_controller)

        self.ui.step0_input_path_button.clicked.connect(self.choose_input)
        self.ui.step0_import_raw_button.clicked.connect(self.load_raw)
        self.ui.step0_load_chan_loc_button.clicked.connect(self.load_channel_location)
        self.ui.step0_save_path_button.clicked.connect(self.new_study_save_path)

        self.ui.step0_no_option_checkbox.clicked.connect(self.newstudy_controller)
        self.ui.step0_filter_option_checkbox.clicked.connect(self.newstudy_controller)
        self.ui.step0_downsamp_option_checkbox.clicked.connect(self.newstudy_controller)
        self.ui.step0_preprocess_data_button.clicked.connect(self.preprocess_data)


        self.ui.step0_selected_files_list.itemClicked.connect(self.plot_CHANNELS)
        self.ui.step0_selected_files_list.itemClicked.connect(self.plot_PSD)
        self.ui.rawdata_plot_button.clicked.connect(self.plot_EEG)
        self.ui.step0_remove_file_button.clicked.connect(self.remove_file)
        self.ui.step0_clear_files_button.clicked.connect(self.clear_files)


    def newstudy_controller(self):
        # widget: Load files with pattern
        if self.ui.step0_load_pattern_radio.isChecked():
            self.ui.step0_import_pattern_lineedit.setEnabled(True)
        else:
            self.ui.step0_import_pattern_lineedit.setDisabled(True)

        if self.ui.step0_input_path_lineedit:
            self.input_folder_found = True
        else:
            self.input_folder_found = False

        if self.input_folder_found:
            self.ui.step0_import_raw_button.setEnabled(True)
            if self.ui.step0_selected_files_list.count() == 0:
                self.data_found = False
                self.ui.MplWidget_chan.canvas.axes.clear()
                self.ui.MplWidget_chan.canvas.draw()
                self.ui.MplWidget_psd.canvas.axes.clear()
                self.ui.MplWidget_psd.canvas.draw()
            else:
                self.data_found = True


        else:
            self.ui.step0_import_raw_button.setDisabled(True)

        plot_options = [
        self.ui.rawdata_plot_button,
        self.ui.rawdata_show_channel_names_checkbox,
        self.ui.rawdata_range_psd_label,
        self.ui.rawdata_range_psd_min_label,
        self.ui.rawdata_range_psd_max_label,
        self.ui.rawdata_range_psd_min,
        self.ui.rawdata_range_psd_max,
        self.ui.rawdata_range_hz1,
        self.ui.rawdata_range_hz2
        ]

        preprocessing_options = [
        self.ui.step0_no_option_checkbox,
        self.ui.step0_filter_option_checkbox,
        self.ui.step0_downsamp_option_checkbox,
        self.ui.step0_preprocess_data_button,
        self.ui.step0_ch2rm_label,
        self.ui.step0_ch2rm_radio,
        self.ui.step0_ch2rm_missing_radio,
        self.ui.step0_ch2rm_input
        ]

        preprocessing_sub_options = [
        self.ui.step0_filter_method_label,
        self.ui.step0_fir_filtermethod_radio,
        self.ui.step0_iir_filtermethod_radio,
        self.ui.step0_lowcut_freq_label,
        self.ui.step0_lowcut_freq_input,
        self.ui.step0_filt_hz1,
        self.ui.step0_highcut_freq_label,
        self.ui.step0_highcut_freq_input,
        self.ui.step0_filt_hz2
        ]

        downsample_sub_options = [
        self.ui.step0_downsamp_freq_label,
        self.ui.step0_downsamp_freq_input,
        self.ui.step0_downsamp_hz
        ]
        if self.data_found and self.ui.step0_study_name_lineedit.text() and self.ui.step0_save_path_lineedit.text():
            self.ui.step0_import_raw_button.setStyleSheet("background-color: lightgreen")
            self.ui.step0_remove_file_button.setEnabled(True)
            self.ui.step0_clear_files_button.setEnabled(True)
            self.ui.step0_load_chan_loc_button.setEnabled(True)
            # Enable Plot Options
            set_widgets_status(plot_options, enable=True)
            # Enable Preprocessing Options
            set_widgets_status(preprocessing_options, enable=True)

            if self.ui.step0_no_option_checkbox.isChecked():
                self.ui.step0_preprocessing_progress.setEnabled(True)

                self.ui.step0_filter_option_checkbox.setChecked(False)
                self.ui.step0_filter_option_checkbox.setDisabled(True)
                self.ui.step0_lowcut_freq_input.setDisabled(True)
                self.ui.step0_highcut_freq_input.setDisabled(True)
                self.ui.step0_fir_filtermethod_radio.setDisabled(True)
                self.ui.step0_iir_filtermethod_radio.setDisabled(True)
                self.ui.step0_downsamp_option_checkbox.setChecked(False)
                self.ui.step0_downsamp_option_checkbox.setDisabled(True)
                self.ui.step0_downsamp_freq_input.setDisabled(True)
            else:
                self.ui.step0_filter_option_checkbox.setEnabled(True)
                self.ui.step0_downsamp_option_checkbox.setEnabled(True)
                if self.ui.step0_filter_option_checkbox.isChecked():
                    self.ui.step0_preprocessing_progress.setEnabled(True)
                    self.filter_data = True
                    set_widgets_status(preprocessing_sub_options, enable=True)
                else:
                    self.filter_data = False
                    set_widgets_status(preprocessing_sub_options, enable=False)
                if self.ui.step0_downsamp_option_checkbox.isChecked():
                    self.ui.step0_preprocessing_progress.setEnabled(True)
                    self.ui.downsample_data = True
                    set_widgets_status(downsample_sub_options, enable=True)
                else:
                    self.downsample_data = False
                    set_widgets_status(downsample_sub_options, enable=False)
        else:
            # Disable Next Steps
            self.ui.step0_import_raw_button.setStyleSheet("background-color: light gray")
            self.ui.step0_remove_file_button.setDisabled(True)
            self.ui.step0_clear_files_button.setDisabled(True)
            self.ui.step0_load_chan_loc_button.setDisabled(True)
            # Enable Plot Options
            set_widgets_status(plot_options, enable=False)
            # Enable Preprocessing Options
            set_widgets_status(preprocessing_options, enable=False)


    def choose_input(self):
        fname = QFileDialog.getExistingDirectory(self, "Select the folder containing raw data")
        self.input_folder = fname
        self.ui.step0_input_path_lineedit.setText(fname)
        self.newstudy_controller()

    def load_channel_location(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select the file containing the channel locations")
        chan_loc_extension = os.path.split(fname)[1].split('.')[1]
        valid_chan_loc_extensions = ['loc', 'locs', 'eloc', 'sfp', 'csd', 'elc', 'txt',
                                     'csd', 'elp', 'bvef', 'csv', 'tsv', 'xyz']
        if not chan_loc_extension in valid_chan_loc_extensions:
            self.channel_location_dir = ''
            QMessageBox.information(self, "Load error",
                                    "File extension is expected to be: ‘.loc’ or ‘.locs’ or ‘.eloc’ (for EEGLAB files),"
                                    "‘.sfp’ (BESA/EGI files), ‘.csd’, ‘.elc’, ‘.txt’, ‘.csd’, ‘.elp’ (BESA spherical),"
                                    "‘.bvef’ (BrainVision files), ‘.csv’, ‘.tsv’, ‘.xyz’ (XYZ coordinates)",
                                    QMessageBox.Ok)
        else:
            self.channel_location_dir = fname
            self.ui.step0_load_chan_loc_button.setStyleSheet("background-color: lightgreen")
        self.newstudy_controller()

    def get_extension(self):
        selected_extension = self.ui.step0_import_format_combobox.currentText()
        extension = selected_extension.split('(')[1]
        extension = re.split(', | .', extension)[0]
        return extension

    def get_data_type(self):
        if self.ui.step0_import_continuous_radio.isChecked():
            data_type = "continuous"
        elif self.ui.step0_import_epoched_radio.isChecked():
            data_type = "epoched"
        else:
            raise ValueError("Failed to match data_type")
        return data_type

    def load_raw(self):
        self.ui.step0_selected_files_list.clear()
        if self.ui.step0_load_all_radio.isChecked():
            self.pattern = '*'
        if self.ui.step0_load_pattern_radio.isChecked():
            self.pattern = '*'+self.ui.step0_import_pattern_lineedit.text()+'*'
        self.extension = self.get_extension()
        self.data_type = self.get_data_type()
        self.list_eegs = find_data(self.input_folder, self.extension, self.pattern)
        for i in range(len(self.list_eegs)):
            self.ui.step0_selected_files_list.addItem(str(self.list_eegs[i]))
        # self.ui.foldername_preprocessed_data = os.path.join(self.ui.input_folder, 'output')
        self.newstudy_controller()

    def new_study_save_path(self):
        #step0_study_name_linedit.text()
        path = QFileDialog.getExistingDirectory(self, "Select the folder to save results")
        folder_name = self.ui.step0_study_name_lineedit.text()
        save_directory = os.path.join(path, folder_name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        else:
            reply = QMessageBox.question(self, "Study exists!",
                                         "Do you want to overwrite an existing study?",
                                         QMessageBox.Yes |
                                         QMessageBox.No)
            if reply == QMessageBox.Yes:
                shutil.rmtree(save_directory)
                os.makedirs(save_directory)
            else:
                folder_name = folder_name+'_new'
                self.ui.step0_study_name_lineedit.setText(folder_name)
                save_directory = os.path.join(path, folder_name)

        self.study_name = self.ui.step0_study_name_lineedit.text()
        self.save_dir = save_directory
        self.ui.step0_save_path_lineedit.setText(save_directory)
        self.use_raw_data = True
        self.newstudy_controller()

    def remove_file(self):
        listItems = self.step0_selected_files_list.selectedItems()
        if not listItems: return
        for item in listItems:
            # To remove items from the list, use takeItem() .
            self.step0_selected_files_list.takeItem(self.step0_selected_files_list.row(item))
        self.newstudy_controller()

    def clear_files(self):
        self.ui.step0_selected_files_list.clear()
        self.newstudy_controller()

    def preprocess_data(self):

        # Initialize config
        config_file = os.path.join(self.save_dir, 'log.ini')
        config = load_config(config_file)
        initialize_config(config_file, config)

        self.save_preprocessed_path = os.path.join(self.save_dir, 'preprocessed_data')
        if not os.path.exists(self.save_preprocessed_path):
            os.makedirs(self.save_preprocessed_path)

        if self.ui.step0_no_option_checkbox.isChecked():
            self.filter_data = False
            self.lowcut_freq = ''
            self.highcut_freq = ''
            self.downsample_data = False
            self.sample_rate = ''

        if self.ui.step0_filter_option_checkbox.isChecked():
            self.filter_data = True
            if self.ui.step0_lowcut_freq_input.text() >= self.ui.step0_highcut_freq_input.text():
                QMessageBox.information(self, "Filter Error",
                                        "Please modify the filter range!",
                                        QMessageBox.Ok)
            if self.ui.step0_fir_filtermethod_radio.isChecked():
                self.filter_method = 'fir'
            elif self.ui.step0_iir_filtermethod_radio.isChecked():
                self.filter_method = 'iir'

            self.lowcut_freq = int(self.ui.step0_lowcut_freq_input.text())
            self.highcut_freq = int(self.ui.step0_highcut_freq_input.text())

        else:
            self.filter_data = False
            self.filter_method = ''
            self.lowcut_freq = ''
            self.highcut_freq = ''

        if self.ui.step0_downsamp_option_checkbox.isChecked():
            self.sample_rate = int(self.ui.step0_downsamp_freq_input.text())
        else:
            self.downsample_data = False
            self.sample_rate = ''

        if self.ui.step0_ch2rm_radio.isChecked():
            self.ch2rm = self.ui.step0_ch2rm_input.text()
            print(self.ch2rm)
        elif self.ui.step0_ch2rm_missing_radio.isChecked():
            self.ch2rm = 'missing'

        print("preprocessing data ...")

        # Write "study info" to config
        config['study info']['study_name'] = self.ui.step0_study_name_lineedit.text()
        config['study info']['input_folder'] = self.ui.step0_input_path_lineedit.text()
        config['study info']['input_data_extension'] = self.extension
        config['study info']['input_data_type'] = self.data_type
        config['study info']['input_name_pattern'] = self.pattern
        list_eegs = []
        for eegpath in self.list_eegs:
            eegfilename = os.path.basename(eegpath)
            eegfilename = os.path.splitext(eegfilename)[0]
            list_eegs = np.append(list_eegs, eegfilename)
        list_eegs = ','.join(map(str, list_eegs))
        config['study info']['input_filenames'] = list_eegs
        config['study info']['save_folder'] = self.save_dir
        save_config(config_file, config)

        length_all_data = []

        counter = 1
        for filename in self.list_eegs:
            self.progress, preprocessed_data, length_data, eeg_info, channels2remove = preprocess.preprocess_eegs(
                filename,
                self.list_eegs,
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
            eeg_info_path = os.path.join(self.save_dir, "eeg_info.pkl")
            save_eeg_info(eeg_info_path, eeg_info)

            self.ch_names, ch_location = list(eeg_info['ch_names']), eeg_info['chs']
            self.n_chan = len(self.ch_names)
            name = os.path.basename(filename)
            name = os.path.splitext(name)[0]
            save_path = os.path.join(self.save_preprocessed_path, name + ".hdf")
            with h5py.File(save_path, "w") as hf:
                dataset = hf.create_dataset(name, data=preprocessed_data, compression="gzip", compression_opts=9)
                # add metadata
                dataset.attrs['data_length'] = preprocessed_data.shape[1]
                dataset.attrs['eeg_format'] = self.extension
                dataset.attrs['data_type'] = self.data_type
                dataset.attrs['nchan'] = self.n_chan
                dataset.attrs['ch_names'] = self.ch_names
                dataset.attrs['ch_removed'] = self.ch2rm
                dataset.attrs['sample_rate'] = self.sample_rate
                dataset.attrs['filter_method'] = self.filter_method
                dataset.attrs['lowcut_freq'] = self.lowcut_freq
                dataset.attrs['highcut_freq'] = self.highcut_freq
            
            # export_h5.export_h5(preprocessed_data, filename, self.ch_names, self.extension, self.data_type,
            #                    self.filter_method, self.lowcut_freq, self.highcut_freq,
            #                    self.sample_rate, self.ch2rm, self.save_preprocessed_path)

            if counter == 1:
                filenames = filename
                catdata = preprocessed_data
            else:
                filenames = np.append(filenames, filename)
                catdata = np.append(catdata, preprocessed_data, axis=1)
            counter += 1

            # Save concat data
            '''
            hf = h5py.File(os.path.join(self.save_preprocessed_path, self.study_name + '_data.hdf'), 'w')
            group = hf.create_group(self.study_name)
            print('\nConcatenating EEG Data ... ', filename)
            group.create_dataset(name, data=preprocessed_data, compression="gzip", compression_opts=9)
            # add metadata
            hf.attrs['data_length'] = preprocessed_data.shape[1]
            hf.attrs['eeg_format'] = self.extension
            hf.attrs['data_type'] = self.data_type
            hf.attrs['nchan'] = self.n_chan
            hf.attrs['ch_names'] = self.ch_names
            hf.attrs['ch_removed'] = self.ch2rm
            hf.attrs['sample_rate'] = self.sample_rate
            hf.attrs['filter_method'] = self.filter_method
            hf.attrs['lowcut_freq'] = self.lowcut_freq
            hf.attrs['highcut_freq'] = self.highcut_freq
            hf.close()
            '''

            length_all_data = np.append(length_all_data, int(length_data))
            print("Progress:", self.progress, "%")

            self.ui.step0_preprocessing_progress.setValue(int(self.progress))
            if self.progress == 100:
                self.done_preprocessing = True
                print("\nSaving the concatenated data ...")
                # Save catdata
                catdata_filename = os.path.join(self.save_dir, self.study_name + "_concatenated_data.hdf")
                with h5py.File(catdata_filename, "w") as catf:
                    catf.create_dataset(self.study_name, data=catdata, compression="gzip", compression_opts=9)

                # Write logs to config
                config.set('progress', 'done_preprocessing', str(self.done_preprocessing))
                # Write "preprocessing settings" to config
                config['preprocessing settings']['filter_data'] = str(self.filter_data)
                config['preprocessing settings']['filter_method'] = str(self.filter_method)
                config['preprocessing settings']['lowcut_freq'] = str(self.lowcut_freq)
                config['preprocessing settings']['highcut_freq'] = str(self.highcut_freq)
                config['preprocessing settings']['downsample_data'] = str(self.downsample_data)
                config['preprocessing settings']['sample_rate'] = str(self.sample_rate)
                channels2remove = ','.join(map(str, channels2remove))
                config['preprocessing settings']['channels2remove'] = channels2remove
                # Write "preprocessing results" to config
                length_data_str = ','.join(map(str, length_all_data))
                config['preprocessing results']['length_data'] = length_data_str
                ch_names = ','.join(map(str, self.ch_names))
                config['preprocessing results']['n_chan'] = str(self.n_chan)
                config['preprocessing results']['ch_names'] = ch_names
                save_config(config_file, config)

                # call mainwindow.load_study()
                if self.main_window:
                    self.main_window.load_study(self.save_dir)
                self.ui.close()

    def plot_CHANNELS(self):
        filename = self.ui.step0_selected_files_list.currentItem().text()
        EEG = load_eegs(filename, self.extension, self.data_type, self.channel_location_dir, [])
        ax = self.ui.MplWidget_chan.canvas.axes
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        if self.channel_location_dir:
            montage = mne.channels.read_custom_montage(self.channel_location_dir)
            EEG.set_montage(montage)
        if np.isnan(EEG.info['chs'][0]['loc'][0]):
            print('No valid channel positions found!')
        else:
            if self.ui.rawdata_show_channel_names_checkbox.isChecked():
                show_names = True
            else:
                show_names = False
            EEG.plot_sensors(ch_type='eeg', show_names=show_names, axes=ax)
        self.ui.MplWidget_chan.canvas.draw()
    
    def plot_EEG(self):
        filename = self.ui.step0_selected_files_list.currentItem().text()
        EEG = load_eegs(filename, self.extension, self.data_type, self.channel_location_dir, [])
        EEG.plot()

    def plot_PSD(self):
        filename = self.ui.step0_selected_files_list.currentItem().text()
        EEG = load_eegs(filename, self.extension, self.data_type, self.channel_location_dir, [])
        if self.ui.step0_filter_option_checkbox.isChecked():
            lowcut = int(self.ui.step0_lowcut_freq_input.text())
            highcut = int(self.ui.step0_highcut_freq_input.text())
            if self.ui.step0_fir_filtermethod_radio.isChecked():
                filter_method = 'fir'
            elif self.ui.step0_iir_filtermethod_radio.isChecked():
                filter_method = 'iir'
            EEG = EEG.filter(l_freq=lowcut, h_freq=highcut, method=filter_method, n_jobs=-1)
        fmin_plot = int(self.ui.rawdata_range_psd_min.text())
        fmax_plot = int(self.ui.rawdata_range_psd_max.text())
        ax = self.ui.MplWidget_psd.canvas.axes
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        EEG.plot_psd(fmin=fmin_plot, fmax=fmax_plot, ax=ax)
        self.ui.MplWidget_psd.canvas.draw()
