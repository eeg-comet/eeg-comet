
import os.path
import re
import numpy as np
import mne
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QDialog, QComboBox, QMessageBox, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from gui_utils.CheckableComboBox import CheckableComboBox
from gui_utils.set_widgets_status import set_widgets_status
from data_utils.data_io import DataIO


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
        self.ui.step0_template_montage_combobox.setCurrentText("standard_1020")
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
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
            self.ui.step0_iclabel_option_checkbox,
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
        [button.activated.connect(action) if isinstance(button, QComboBox) else button.clicked.connect(action) for
         button, action in buttons_actions]

    def newstudy_controller(self):
        """
        Control the behavior of the New Study window based on user selections.
        """
        common_message = f"Load {self.get_data_type()} EEG data with {self.get_extension()} extension"
        if self.ui.step0_load_all_radio.isChecked():
            self.ui.step0_import_log_lineedit.setText(f"{common_message} all.")
            self.ui.step0_import_pattern_lineedit.clear()
        elif self.ui.step0_load_pattern_radio.isChecked() and self.ui.step0_import_pattern_lineedit.text().strip():
            pattern = self.ui.step0_import_pattern_lineedit.text()
            self.ui.step0_import_log_lineedit.setText(f"{common_message} that contain '{pattern}' in their filenames.")
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
            self.ui.step0_iclabel_option_checkbox,
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
                widgets_to_enable = [self.ui.step0_import_raw_button, self.ui.step0_remove_file_button,
                                     self.ui.step0_clear_files_button] + preprocessing_widgets
                set_widgets_status(widgets_to_enable, mode='enable')
            else:
                widgets_to_disable = [self.ui.step0_remove_file_button,
                                      self.ui.step0_clear_files_button] + plot_widgets + preprocessing_widgets
                set_widgets_status(widgets_to_disable, mode='disable')
        if self.ui.step0_selected_files_list.count() != 0:
            set_widgets_status(self.ui.step0_preprocess_radio, mode='enable')
            if self.ui.step0_selected_files_list.currentItem():
                set_widgets_status(plot_widgets, mode='enable')
        else:
            set_widgets_status(self.ui.step0_preprocess_radio, mode='disable')
        if self.ui.step0_preprocess_radio.isChecked():
            widgets_to_show = preprocessing_widgets + filter_sub_widgets + downsample_sub_widgets
            widgets_to_hide_or_disable = [self.ui.step0_import_raw_button] + import_data_widgets
            set_widgets_status(widgets_to_show, mode='show')
            set_widgets_status(widgets_to_hide_or_disable, mode='disable')
            set_widgets_status(widgets_to_hide_or_disable, mode='hide')
            widget_conditions = [
                (self.ui.step0_template_montage_combobox, self.ui.step0_use_template_montage_radio.isChecked()),
                (self.ui.step0_chanloc_path_lineedit, self.ui.step0_load_montage_radio.isChecked()),
                (self.ui.step0_ch2rm_combobox, self.ui.step0_ch2rm_radio.isChecked())
            ]
            for widget, condition in widget_conditions:
                set_widgets_status(widget, 'enable' if condition else 'disable')
            if self.ui.step0_ch2rm_missing_radio.isChecked():
                self.ui.step0_ch2rm_combobox.deselectAllItems()
            self.filter_data = self.ui.step0_filter_option_checkbox.isChecked()
            set_widgets_status(filter_sub_widgets, mode='enable' if self.filter_data else 'disable')
            if not self.filter_data:
                self.lowcut_freq = ''
                self.highcut_freq = ''
            self.downsample_data = self.ui.step0_downsamp_option_checkbox.isChecked()
            set_widgets_status(downsample_sub_widgets, mode='enable' if self.downsample_data else 'disable')
            set_widgets_status(self.ui.step0_preprocess_data_button,
                               'enable' if self.ui.step0_save_path_lineedit.text() else 'disable')
            self.iclabel_data = self.ui.step0_iclabel_option_checkbox.isChecked()

    def choose_input(self):
        """
        Open a file dialog to select the folder containing raw data
        """
        self.input_folder = QFileDialog.getExistingDirectory(self, "Select the folder containing raw data")
        self.ui.step0_input_path_lineedit.setText(self.input_folder)
        self.newstudy_controller()

    def load_custom_montage(self):
        """
        Load a custom montage file and set the channel location directory.
        """
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setWindowTitle("Select the file(s) containing the channel locations")
        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
            for fname in file_names:
                chan_loc_extension = os.path.basename(fname).split('.')[-1]
                valid_chan_loc_extensions = ['loc', 'locs', 'eloc', 'sfp', 'csd', 'elc', 'txt',
                                             'csd', 'elp', 'bvef', 'csv', 'tsv', 'xyz']
                if chan_loc_extension not in valid_chan_loc_extensions:
                    QMessageBox.information(
                        self, "Load error",
                        "File extension is expected to be: ‘.loc’ or ‘.locs’ or ‘.eloc’ (for EEGLAB files),"
                        "‘.sfp’ (BESA/EGI files), ‘.csd’, ‘.elc’, ‘.txt’, ‘.csd’, ‘.elp’ (BESA spherical),"
                        "‘.bvef’ (BrainVision files), ‘.csv’, ‘.tsv’, ‘.xyz’ (XYZ coordinates)", QMessageBox.Ok)
                    self.comet_tbx.channel_location_dir = ''
                    return
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
        return "epoched" if self.ui.step0_import_epoched_radio.isChecked() else "raw"

    def update_channel_names(self):
        """
        Update the channel names based on the selected EEG file.
        """
        filename = self.ui.step0_selected_files_list.currentItem().text()
        eeg = DataIO().load_eegs(filename, self.comet_tbx.datatype, self.comet_tbx.channel_location_dir, [])
        data_channel_names = eeg.info['ch_names']
        self.ui.step0_ch2rm_combobox.addItems(data_channel_names)
        montage = None
        if not np.isnan(eeg.info['chs'][0]['loc'][0]):
            montage = eeg.get_montage()
        elif self.ui.step0_load_montage_radio.isChecked() and os.path.isfile(
                self.ui.step0_chanloc_path_lineedit.text()):
            montage = mne.channels.read_custom_montage(self.ui.step0_chanloc_path_lineedit.text())
        elif self.ui.step0_use_template_montage_radio.isChecked():
            if not self.comet_tbx.channel_location_dir:
                self.comet_tbx.channel_location_dir = "standard_1020"
            montage = mne.channels.make_standard_montage(self.comet_tbx.channel_location_dir)
        eeg.set_montage(montage, match_case=False, on_missing='warn')

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
        self.ui.step0_import_log_lineedit.setText(
            f"{len(self.comet_tbx.list_eegs_path)} EEG data were detected."
        )
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
        selected_items = self.step0_selected_files_list.selectedItems()
        if not selected_items:
            return
        [self.step0_selected_files_list.takeItem(self.step0_selected_files_list.row(item)) for item in selected_items]
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
        self.preprocessed_data_path = os.path.join(
            self.save_dir, f'{self.study_name}_preprocessed_data'
        )
        self.filter_data = self.ui.step0_filter_option_checkbox.isChecked()
        if self.filter_data:
            if self.ui.step0_lowcut_freq_input.text() >= self.ui.step0_highcut_freq_input.text():
                QMessageBox.information(self, "Filter Error",
                                        "Please modify the filter range!",
                                        QMessageBox.Ok)
            self.filter_method = 'fir' if self.ui.step0_fir_filtermethod_radio.isChecked() else 'iir'
            self.lowcut_freq = int(self.ui.step0_lowcut_freq_input.text())
            self.highcut_freq = int(self.ui.step0_highcut_freq_input.text())
        else:
            self.filter_method = ''
            self.lowcut_freq = ''
            self.highcut_freq = ''
        self.downsample_data = self.ui.step0_downsamp_option_checkbox.isChecked()
        self.sample_rate = int(self.ui.step0_downsamp_freq_input.text()) if self.downsample_data else ''
        self.iclabel_data = self.ui.step0_iclabel_option_checkbox.isChecked()
        self.chan2rm = (self.ui.step0_ch2rm_combobox.currentData() if self.ui.step0_ch2rm_radio.isChecked()
                        else 'missing')
        attributes = ['preprocessed_data_path', 'filter_data', 'filter_method', 'lowcut_freq',
                      'highcut_freq', 'downsample_data', 'sample_rate', 'chan2rm', 'save_dir']
        for attr in attributes:
            setattr(self.comet_tbx, attr, getattr(self, attr))
        self.comet_tbx.study_name = self.ui.step0_study_name_lineedit.text()
        self.comet_tbx.eeg_info_path = os.path.join(self.comet_tbx.save_dir, "eeg_info.pkl")
        self.comet_tbx.do_preprocessing()
        self.done_preprocessing = True
        self.comet_tbx.save_tbx()
        if self.main_window:
            self.main_window.load_study(from_new_study=True)
        self.ui.close()

    def plot_montage(self):
        """
        Plot EEG montage and display channel names if selected.
        """
        self.canvas.figure.clear()
        filename = self.ui.step0_selected_files_list.currentItem().text()
        eeg = DataIO().load_eegs(filename, self.comet_tbx.datatype, self.comet_tbx.channel_location_dir, [])
        if eeg.info['dig'] is None:
            QMessageBox.information(self, "Load error",
                                    "Unable to retrieve channel locations."
                                    "Please ensure they are imported before proceeding.",
                                    QMessageBox.Ok)
        else:
            show_names = self.ui.rawdata_show_channel_names_checkbox.isChecked()
            if self.ui.step0_ch2rm_combobox.currentData():
                self.chan2rm = self.ui.step0_ch2rm_combobox.currentData()
                eeg.info["bads"].extend(self.chan2rm)
            fig, _ = eeg.plot_sensors(kind='select', show_names=show_names, show=False)
            self.ui.figure_title_lineedit.setText("EEG Montage")
            self.canvas.figure = fig
            self.canvas.draw()

    def plot_eeg(self):
        """
        Plot the EEG data.
        """
        filename = self.ui.step0_selected_files_list.currentItem().text()
        eeg = DataIO().load_eegs(filename, self.comet_tbx.datatype, self.comet_tbx.channel_location_dir)
        eeg.plot()

    def plot_psd(self):
        """
        Plot the Power Spectral Density (PSD) of EEG data.
        """
        self.canvas.figure.clear()
        filename = self.ui.step0_selected_files_list.currentItem().text()
        eeg = DataIO().load_eegs(filename, self.comet_tbx.extension, self.comet_tbx.datatype,
                                 self.comet_tbx.channel_location_dir, [])
        if self.ui.step0_filter_option_checkbox.isChecked():
            lowcut = int(self.ui.step0_lowcut_freq_input.text())
            highcut = int(self.ui.step0_highcut_freq_input.text())
            filter_method = 'iir' if self.ui.step0_iir_filtermethod_radio.isChecked() else 'fir'
            eeg = eeg.filter(l_freq=lowcut, h_freq=highcut, method=filter_method, n_jobs=-1)
        fmin_plot = int(self.ui.rawdata_range_psd_min.text())
        fmax_plot = int(self.ui.rawdata_range_psd_max.text())
        fig = eeg.compute_psd(fmin=fmin_plot, fmax=fmax_plot).plot(show=False)
        self.canvas.figure = fig
        self.ui.figure_title_lineedit.setText("Power Spectral Density (PSD) using Multitapers")
        self.canvas.draw()
