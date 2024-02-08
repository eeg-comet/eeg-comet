
import os.path
import re
import numpy as np
import mne
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QDialog, QComboBox, QMessageBox, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from functions.gui_utils.CheckableComboBox import CheckableComboBox
from functions.gui_utils.set_widgets_status import set_widgets_status
from functions.data_utils.data_io import DataIO


class NewStudyWindow(QDialog):
    def __init__(self, context, parent=None, main_window=None, comet_tbx=None):
        super().__init__(parent)
        self.main_window = main_window
        self.comet_tbx = comet_tbx
        self.ui = uic.loadUi(context.get_resource("NewStudyWindow.ui"), self)
        self.ui.setWindowTitle("New Study - Import EEG Data and Preprocess")
        self.done_preprocessing = False
        self.init_ui_components()
        self.setup_connections()
        self.comet_tbx.channel_location_dir = ""
        self.newstudy_controller()

    def init_ui_components(self):
        """
        Initialize UI components.
        """
        self.ui.step0_ch2rm_combobox = CheckableComboBox()
        self.CheckableComboBox_Layout.addWidget(self.ui.step0_ch2rm_combobox)
        builtin_montages = mne.channels.get_builtin_montages()
        self.ui.step0_template_montage_combobox.addItems(builtin_montages)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.ui.Figure_Layout.addWidget(self.canvas)

    def setup_connections(self):
        """
        Set up signal-slot connections.
        """
        # Controlling the visibility and state of various UI components based on user interactions
        control_items = [
            self.ui.step0_import_data_radio,
            self.ui.step0_preprocess_radio,
            self.ui.step0_import_epoched_radio,
            self.ui.step0_import_raw_radio,
            self.ui.step0_load_all_radio,
            self.ui.step0_load_pattern_radio,
            self.ui.step0_load_montage_radio,
            self.ui.step0_use_template_montage_radio,
            self.ui.step0_filter_option_checkbox,
            self.ui.step0_downsamp_option_checkbox,
            self.ui.step0_ch2rm_radio,
            self.ui.step0_ch2rm_missing_radio,
        ]
        for item in control_items:
            item.clicked.connect(self.newstudy_controller)
        self.ui.step0_study_name_lineedit.textChanged.connect(self.newstudy_controller)
        self.ui.step0_import_pattern_lineedit.textChanged.connect(self.newstudy_controller)
        self.ui.step0_selected_files_list.itemClicked.connect(self.newstudy_controller)
        self.ui.step0_selected_files_list.itemClicked.connect(self.update_channel_names)
        # Button connections for performing specific tasks
        buttons_actions = [
            (self.ui.step0_input_path_button, self.choose_input),
            (self.ui.step0_import_raw_button, self.load_raw),
            (self.ui.step0_load_montage_radio, self.load_custom_montage),
            (self.ui.step0_template_montage_combobox, self.load_template_montage),
            (self.ui.step0_save_path_button, self.new_study_save_path),
            (self.ui.step0_preprocess_data_button, self.preprocess_data),
            (self.ui.step0_remove_file_button, self.remove_file),
            (self.ui.step0_clear_files_button, self.clear_files),
            (self.ui.show_montage_button, self.plot_montage),
            (self.ui.rawdata_show_channel_names_checkbox, self.plot_montage),
            (self.ui.step0_ch2rm_combobox, self.plot_montage),
            (self.ui.show_psd_button, self.plot_psd),
            (self.ui.rawdata_plot_button, self.plot_eeg),
        ]
        for button, action in buttons_actions:
            if isinstance(button, QComboBox):
                button.activated.connect(action)
            else:
                button.clicked.connect(action)

    def newstudy_controller(self):
        """
        Control the behavior of the New Study window based on user selections.
        """

        if self.ui.step0_load_all_radio.isChecked():
            self.ui.step0_import_log_lineedit.setText(
                f"Load all {self.get_data_type()} EEG data with {self.get_extension()} extension.")
            self.ui.step0_import_pattern_lineedit.clear()
        if self.ui.step0_load_pattern_radio.isChecked() and self.ui.step0_import_pattern_lineedit.text().strip():
            self.ui.step0_import_log_lineedit.setText(
                f"Load {self.get_data_type()} EEG data with {self.get_extension()} extension that contain "
                f"'{self.ui.step0_import_pattern_lineedit.text()}' in their filenames."
            )

        plot_widgets = [
            self.ui.rawdata_plot_button,
            self.ui.show_montage_button,
            self.ui.show_psd_button,
            self.ui.rawdata_show_channel_names_checkbox,
            self.ui.rawdata_range_psd_label,
            self.ui.rawdata_range_psd_min_label,
            self.ui.rawdata_range_psd_max_label,
            self.ui.rawdata_range_psd_min,
            self.ui.rawdata_range_psd_max,
            self.ui.rawdata_range_hz1,
            self.ui.rawdata_range_hz2
        ]

        import_data_widgets = [
            self.ui.step0_input_path_label,
            self.ui.step0_study_name_label,
            self.ui.step0_input_path_button,
            self.ui.step0_import_format_label,
            self.ui.step0_import_type_label,
            self.ui.step0_import_pattern_label,
            self.ui.step0_study_name_lineedit,
            self.ui.step0_input_path_lineedit,
            self.ui.step0_import_format_combobox,
            self.ui.step0_import_raw_radio,
            self.ui.step0_import_epoched_radio,
            self.ui.step0_load_all_radio,
            self.ui.step0_load_pattern_radio,
            self.ui.step0_import_log_lineedit
        ]

        preprocessing_widgets = [
            self.ui.step0_montage_label,
            self.ui.step0_load_montage_radio,
            self.ui.step0_chanloc_path_lineedit,
            self.ui.step0_use_template_montage_radio,
            self.step0_template_montage_combobox,
            self.ui.step0_filter_option_checkbox,
            self.ui.step0_downsamp_option_checkbox,
            self.ui.step0_ch2rm_label,
            self.ui.step0_ch2rm_radio,
            self.ui.step0_ch2rm_combobox,
            self.ui.step0_ch2rm_missing_radio,
            self.ui.step0_save_path_button,
            self.ui.step0_save_path_lineedit,
            self.ui.step0_preprocess_data_button
        ]

        filter_sub_widgets = [
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

        downsample_sub_widgets = [
            self.ui.step0_downsamp_freq_label,
            self.ui.step0_downsamp_freq_input,
            self.ui.step0_downsamp_hz
        ]

        if self.ui.step0_import_data_radio.isChecked():
            widgets_to_rm = (
                    preprocessing_widgets +
                    filter_sub_widgets +
                    downsample_sub_widgets
            )
            set_widgets_status(import_data_widgets, mode='show')
            set_widgets_status(import_data_widgets, mode='enable')
            set_widgets_status(self.ui.step0_import_raw_button, mode='show')
            set_widgets_status(widgets_to_rm, mode='disable')
            set_widgets_status(widgets_to_rm, mode='hide')

            set_widgets_status(
                self.ui.step0_import_pattern_lineedit,
                'enable' if self.ui.step0_load_pattern_radio.isChecked()
                else 'disable')

            if (self.ui.step0_study_name_lineedit.text()
                    and self.ui.step0_input_path_lineedit):

                set_widgets_status(self.ui.step0_import_raw_button, mode='enable')
                self.ui.step0_remove_file_button.setEnabled(True)
                self.ui.step0_clear_files_button.setEnabled(True)
                # Enable Preprocessing Options
                set_widgets_status(preprocessing_widgets, mode='enable')

            else:
                # Disable Next Steps
                self.ui.step0_remove_file_button.setDisabled(True)
                self.ui.step0_clear_files_button.setDisabled(True)
                # Disable Plot Options
                set_widgets_status(plot_widgets, mode='disable')
                # Disable Preprocessing Options
                set_widgets_status(preprocessing_widgets, mode='disable')

        if not self.ui.step0_selected_files_list.count() == 0:
            set_widgets_status(self.ui.step0_preprocess_radio, mode='enable')

            # Enable Plot Options
            if self.ui.step0_selected_files_list.currentItem():
                set_widgets_status(plot_widgets, mode='enable')
        else:
            set_widgets_status(self.ui.step0_preprocess_radio, mode='disable')

        if self.ui.step0_preprocess_radio.isChecked():
            widgets_to_show = (
                    preprocessing_widgets +
                    filter_sub_widgets +
                    downsample_sub_widgets
            )
            set_widgets_status(widgets_to_show, mode='show')
            set_widgets_status(self.ui.step0_import_raw_button, mode='disable')
            set_widgets_status(self.ui.step0_import_raw_button, mode='hide')
            set_widgets_status(import_data_widgets, mode='disable')
            set_widgets_status(import_data_widgets, mode='hide')

            set_widgets_status(
                self.ui.step0_template_montage_combobox,
                'enable' if self.ui.step0_use_template_montage_radio.isChecked()
                else 'disable')

            set_widgets_status(
                self.ui.step0_chanloc_path_lineedit,
                'enable' if self.ui.step0_load_montage_radio.isChecked()
                else 'disable')

            set_widgets_status(
                self.ui.step0_ch2rm_combobox,
                'enable' if self.ui.step0_ch2rm_radio.isChecked()
                else 'disable')

            if self.ui.step0_ch2rm_missing_radio.isChecked():
                self.ui.step0_ch2rm_combobox.deselectAllItems()

            if self.ui.step0_filter_option_checkbox.isChecked():
                self.filter_data = True
                set_widgets_status(filter_sub_widgets, mode='enable')
            else:
                self.filter_data = False
                self.lowcut_freq = ''
                self.highcut_freq = ''
                set_widgets_status(filter_sub_widgets, mode='disable')
            if self.ui.step0_downsamp_option_checkbox.isChecked():
                self.ui.downsample_data = True
                set_widgets_status(downsample_sub_widgets, mode='enable')
            else:
                self.downsample_data = False
                set_widgets_status(downsample_sub_widgets, mode='disable')

            set_widgets_status(
                self.ui.step0_preprocess_data_button, 'enable' if self.ui.step0_save_path_lineedit.text()
                else 'disable')

    def choose_input(self):
        """
        Open a file dialog to select the folder containing raw data
        """
        fname = QFileDialog.getExistingDirectory(self, "Select the folder containing raw data")
        self.input_folder = fname
        self.ui.step0_input_path_lineedit.setText(fname)
        self.newstudy_controller()

    def load_custom_montage(self):
        """
        Load a custom montage file and set the channel location directory.
        """
        fname, _ = QFileDialog.getOpenFileName(self, "Select the file containing the channel locations")
        if os.path.isfile(fname):
            chan_loc_extension = os.path.basename(fname).split('.')[-1]
            valid_chan_loc_extensions = ['loc', 'locs', 'eloc', 'sfp', 'csd', 'elc', 'txt',
                                         'csd', 'elp', 'bvef', 'csv', 'tsv', 'xyz']
            if chan_loc_extension not in valid_chan_loc_extensions:
                QMessageBox.information(
                    self, "Load error",
                    "File extension is expected to be: ‘.loc’ or ‘.locs’ or ‘.eloc’ (for EEGLAB files),"
                    "‘.sfp’ (BESA/EGI files), ‘.csd’, ‘.elc’, ‘.txt’, ‘.csd’, ‘.elp’ (BESA spherical),"
                    "‘.bvef’ (BrainVision files), ‘.csv’, ‘.tsv’, ‘.xyz’ (XYZ coordinates)",  QMessageBox.Ok)
                self.comet_tbx.channel_location_dir = ''
            else:
                self.comet_tbx.channel_location_dir = fname
                self.ui.step0_chanloc_path_lineedit.setText(fname)

    def load_template_montage(self):
        """
        Set the channel location using the selected template montage.
        """
        self.comet_tbx.channel_location_dir = self.ui.step0_template_montage_combobox.currentText()

    def get_extension(self):
        """
        Get the selected extension from the data extension combo box.
        """
        selected_extension = self.ui.step0_import_format_combobox.currentText()
        extension = selected_extension.split('(')[1]
        extension = re.split(', | .', extension)[0]
        if extension.endswith(')'):
            extension = extension[:-1]
        return extension

    def get_data_type(self):
        """
        Get the selected data type based on the radio button.
        """
        if self.ui.step0_import_raw_radio.isChecked():
            data_type = "raw"
        elif self.ui.step0_import_epoched_radio.isChecked():
            data_type = "epoched"
        else:
            raise ValueError("Failed to match data_type")
        return data_type

    def update_channel_names(self):
        """
        Update the channel names based on the selected EEG file.
        """
        filename = self.ui.step0_selected_files_list.currentItem().text()
        data_io = DataIO()
        eeg = data_io.load_eegs(filename, self.comet_tbx.extension, self.comet_tbx.datatype,
                                self.comet_tbx.channel_location_dir, [])
        if not np.isnan(eeg.info['chs'][0]['loc'][0]):
            montage = eeg.get_montage()
            channel_names = montage.ch_names
        elif self.ui.step0_load_montage_radio.isChecked() and os.path.isfile(self.ui.step0_chanloc_path_lineedit):
            montage = mne.channels.read_custom_montage(self.comet_tbx.channel_location_dir)
            channel_names = montage.ch_names
        elif self.ui.step0_use_template_montage_radio.isChecked():
            montage = mne.channels.make_standard_montage(self.comet_tbx.channel_location_dir)
            channel_names = montage.ch_names
        self.ui.step0_ch2rm_combobox.addItems(channel_names)

    def load_raw(self):
        """
        Load raw EEG data and update the selected files list.
        """
        self.ui.step0_selected_files_list.clear()

        self.comet_tbx.load_all_files = self.ui.step0_load_all_radio.isChecked()
        self.comet_tbx.pattern_content = self.ui.step0_import_pattern_lineedit.text()
        self.comet_tbx.input_folder = self.input_folder
        self.comet_tbx.extension = self.get_extension()
        self.comet_tbx.datatype = self.get_data_type()
        self.comet_tbx.load_raw()
        for i in range(len(self.comet_tbx.list_eegs_path)):
            self.ui.step0_selected_files_list.addItem(str(self.comet_tbx.list_eegs_path[i]))
        self.ui.step0_import_log_lineedit.setText(f"{str(len(self.comet_tbx.list_eegs_path))} EEG data were detected.")
        if len(self.comet_tbx.list_eegs_path) > 0:
            self.ui.step0_preprocess_radio.setChecked(True)
        self.newstudy_controller()

    def new_study_save_path(self):
        """
        Choose and set the save directory for the new study.
        """
        save_parent_directory = QFileDialog.getExistingDirectory(
            self, "Select a parent folder to create a new study folder within.")
        self.study_name = self.ui.step0_study_name_lineedit.text()

        if not os.path.exists(save_parent_directory):
            print("Unable to find selected directory")
            return

        if not self.study_name:
            self.study_name = "EEG_COMET_NEW_STUDY"
            self.ui.step0_study_name_lineedit.setText(self.study_name)

        save_directory = os.path.join(save_parent_directory, self.study_name)

        if os.path.exists(save_directory):
            QMessageBox.information(self, "A folder with the same study name already exists!",
                                    "Please choose another directory or rename your study.",
                                    QMessageBox.Ok)
        else:
            self.save_dir = save_directory
            self.ui.step0_save_path_lineedit.setText(self.save_dir)

        self.newstudy_controller()

    def remove_file(self):
        """
        Remove the selected files from the selected files list.
        """
        list_items = self.step0_selected_files_list.selectedItems()
        if not list_items: return
        for item in list_items:
            # To remove items from the list, use takeItem() .
            self.step0_selected_files_list.takeItem(self.step0_selected_files_list.row(item))
        self.newstudy_controller()

    def clear_files(self):
        """
        Clear all files from the selected files list.
        """
        self.ui.step0_selected_files_list.clear()
        self.canvas.figure.clear()
        self.newstudy_controller()

    def preprocess_data(self):
        """
        Perform data preprocessing based on user-selected options.
        """
        os.makedirs(self.save_dir)
        self.preprocessed_data_path = os.path.join(self.save_dir, self.study_name+'_preprocessed_data')

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
            self.ch2rm = self.ui.step0_ch2rm_combobox.currentData()
        elif self.ui.step0_ch2rm_missing_radio.isChecked():
            self.ch2rm = 'missing'

        self.comet_tbx.preprocessed_data_path = self.preprocessed_data_path
        self.comet_tbx.filter_data = self.filter_data
        self.comet_tbx.filter_method = self.filter_method
        self.comet_tbx.lowcut_freq = self.lowcut_freq
        self.comet_tbx.highcut_freq = self.highcut_freq
        self.comet_tbx.downsample_data = self.downsample_data
        self.comet_tbx.sample_rate = self.sample_rate
        self.comet_tbx.ch2rm = self.ch2rm
        self.comet_tbx.save_dir = self.save_dir
        self.comet_tbx.study_name = self.ui.step0_study_name_lineedit.text()
        self.comet_tbx.eeg_info_path = os.path.join(self.comet_tbx.save_dir, "eeg_info.pkl")

        list_eegs = []
        for eegpath in self.comet_tbx.list_eegs:
            eegfilename = os.path.basename(eegpath)
            eegfilename = os.path.splitext(eegfilename)[0]
            list_eegs = np.append(list_eegs, eegfilename)
        #list_eegs = ','.join(map(str, list_eegs))

        self.comet_tbx.do_preprocessing()
        self.done_preprocessing = True
        self.comet_tbx.save_tbx()

        if self.main_window:
            self.main_window.tbx = self.comet_tbx

            self.main_window.log_window.append_log(f"Study name: {self.study_name}")
            self.main_window.log_window.append_log(f"Input folder: {self.input_folder}")
            self.main_window.log_window.append_log(
                f"Preprocessed {len(list_eegs)} {self.get_data_type()} EEG data with {self.get_extension()} extension.")
            self.main_window.log_window.append_log(f"Channels removed from data: {self.ch2rm}")
            if self.filter_data:
                self.main_window.log_window.append_log(
                    f"Bandpass filtered data using: {self.filter_method.upper()} "
                    f"Filter within {self.lowcut_freq}Hz and {self.highcut_freq}Hz")
            if self.downsample_data:
                self.main_window.log_window.append_log(f"Downsampled data to: {self.sample_rate}Hz")

            self.main_window.load_study(from_new_study=True)
        self.ui.close()

    def plot_montage(self):
        """
        Plot EEG montage and display channel names if selected.
        """
        self.canvas.figure.clear()
        filename = self.ui.step0_selected_files_list.currentItem().text()
        eeg = DataIO().load_eegs(filename, self.comet_tbx.extension, self.comet_tbx.datatype,
                                 self.comet_tbx.channel_location_dir, [])
        if eeg.info['dig'] is None:
            QMessageBox.information(self, "Load error",
                                    "Unable to retrieve channel locations."
                                    "Please ensure they are imported before proceeding.",
                                    QMessageBox.Ok)
        else:
            #montage = EEG.get_montage()
            if self.ui.rawdata_show_channel_names_checkbox.isChecked():
                show_names = True
            else:
                show_names = False

            if self.ui.step0_ch2rm_combobox.currentData():
                self.ch2rm = self.ui.step0_ch2rm_combobox.currentData()
                eeg.info["bads"].extend(self.ch2rm)

            fig, _ = eeg.plot_sensors(kind='select', show_names=show_names, show=False)
            #fig = montage.plot(show_names=show_names)

            self.ui.figure_title_lineedit.setText("EEG Montage")
            self.canvas.figure = fig
            self.canvas.draw()

    def plot_eeg(self):
        """
        Plot the EEG data.
        """
        filename = self.ui.step0_selected_files_list.currentItem().text()
        eeg = DataIO().load_eegs(filename, self.comet_tbx.extension, self.comet_tbx.datatype,
                                 self.comet_tbx.channel_location_dir)
        eeg.plot()

    def plot_psd(self):
        """
        Plot the Power Spectral Density (PSD) of EEG data.
        """
        self.canvas.figure.clear()
        # self.ui.MplWidget.canvas.draw()
        filename = self.ui.step0_selected_files_list.currentItem().text()
        eeg = DataIO().load_eegs(filename, self.comet_tbx.extension, self.comet_tbx.datatype,
                                 self.comet_tbx.channel_location_dir, [])
        if self.ui.step0_filter_option_checkbox.isChecked():
            lowcut = int(self.ui.step0_lowcut_freq_input.text())
            highcut = int(self.ui.step0_highcut_freq_input.text())
            if self.ui.step0_fir_filtermethod_radio.isChecked():
                filter_method = 'fir'
            elif self.ui.step0_iir_filtermethod_radio.isChecked():
                filter_method = 'iir'
            eeg = eeg.filter(l_freq=lowcut, h_freq=highcut, method=filter_method, n_jobs=-1)
        fmin_plot = int(self.ui.rawdata_range_psd_min.text())
        fmax_plot = int(self.ui.rawdata_range_psd_max.text())
        fig = eeg.compute_psd(fmin=fmin_plot, fmax=fmax_plot).plot(show=False)
        #mne.viz.plot_raw_psd(EEG, fmin=fmin_plot, fmax=fmax_plot, ax=ax)
        self.canvas.figure = fig
        self.ui.figure_title_lineedit.setText("Power Spectral Density (PSD) using Multitapers")
        self.canvas.draw()
