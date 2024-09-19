
import os.path
import re
import numpy as np
from mne.channels import get_builtin_montages, read_custom_montage, make_standard_montage
from mne.viz import plot_topomap
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

    def clean_figure_layout(self):
        """
        Clear and delete all widgets from the figure layout to reset it.
        """
        while self.ui.Figure_Layout.count() > 0:
            item = self.ui.Figure_Layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def init_canvas(self):
        """
        Initialize Canvas, replacing the existing one if it already exists.
        """
        self.clean_figure_layout()
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.ui.Figure_Layout.addWidget(self.canvas)

    def init_ui_components(self):
        """
        Initialize UI components.
        """
        self.ui.step2_ch2rm_combobox = CheckableComboBox()
        self.CheckableComboBox_Layout.addWidget(self.ui.step2_ch2rm_combobox)
        builtin_montages = get_builtin_montages()
        self.ui.step2_template_montage_combobox.addItems(builtin_montages)
        self.ui.step2_template_montage_combobox.setCurrentText("standard_1020")
        self.init_canvas()

    def setup_connections(self):
        """
        Set up signal-slot connections.
        """
        # Controlling the visibility and state of various UI components based on user interactions
        control_items = [
            self.ui.step1_import_data_radio,
            self.ui.step1_import_epoched_radio,
            self.ui.step1_import_raw_radio,
            self.ui.step1_load_all_radio,
            self.ui.step1_load_pattern_radio,
            self.ui.step2_preprocess_radio,
            self.ui.step2_load_montage_radio,
            self.ui.step2_use_template_montage_radio,
            self.ui.step2_filter_option_checkbox,
            self.ui.step2_downsamp_option_checkbox,
            self.ui.step2_ch2rm_radio,
            self.ui.step2_auto_clean_option_checkbox
        ]
        for item in control_items:
            item.clicked.connect(self.newstudy_controller)
        self.ui.step1_study_name_lineedit.textChanged.connect(self.newstudy_controller)
        self.ui.step1_import_pattern_lineedit.textChanged.connect(self.newstudy_controller)
        self.ui.loaded_selected_files_list.itemClicked.connect(self.newstudy_controller)
        self.ui.loaded_selected_files_list.itemClicked.connect(self.update_channel_names)
        # Button connections for performing specific tasks
        buttons_actions = [
            (self.ui.step1_input_path_button, self.choose_input),
            (self.ui.step1_import_raw_button, self.load_raw),
            (self.ui.step2_load_montage_radio, self.load_custom_montage),
            (self.ui.step2_template_montage_combobox, self.load_template_montage),
            (self.ui.step2_save_path_button, self.new_study_save_path),
            (self.ui.step2_preprocess_data_button, self.preprocess_data),
            (self.ui.loaded_remove_file_button, self.remove_file),
            (self.ui.loaded_clear_files_button, self.clear_files),
            (self.ui.vis_montage_button, self.plot_montage),
            (self.ui.vis_channel_names_checkbox, self.plot_montage),
            (self.ui.step2_ch2rm_combobox, self.plot_montage),
            (self.ui.vis_psd_button, self.plot_psd),
            (self.ui.vis_plot_button, self.plot_eeg),
        ]
        [button.activated.connect(action) if isinstance(button, QComboBox) else button.clicked.connect(action) for
         button, action in buttons_actions]

    def newstudy_controller(self):
        """
        Control the behavior of the New Study window based on user selections.
        """
        common_message = f"Load {self.get_data_type()} EEG data with {self.get_extension()} extension"
        if self.ui.step1_load_all_radio.isChecked():
            self.ui.step1_import_log_lineedit.setText(f"{common_message} all.")
            self.ui.step1_import_pattern_lineedit.clear()
        elif self.ui.step1_load_pattern_radio.isChecked() and self.ui.step1_import_pattern_lineedit.text().strip():
            pattern = self.ui.step1_import_pattern_lineedit.text()
            self.ui.step1_import_log_lineedit.setText(f"{common_message} that contain '{pattern}' in their filenames.")
        plot_widgets = [
            self.ui.vis_plot_button,
            self.ui.viz_plot_time_radio,
            self.ui.viz_plot_event_radio,
            self.ui.vis_montage_button,
            self.ui.vis_psd_button,
            self.ui.vis_channel_names_checkbox,
            self.ui.vis_range_psd_label,
            self.ui.vis_range_psd_min_label,
            self.ui.vis_range_psd_max_label,
            self.ui.vis_range_psd_min,
            self.ui.vis_range_psd_max,
            self.ui.vis_range_hz1,
            self.ui.vis_range_hz2
        ]
        import_data_widgets = [
            self.ui.step1_input_path_label,
            self.ui.step1_study_name_label,
            self.ui.step1_input_path_button,
            self.ui.step1_import_format_label,
            self.ui.step1_import_type_label,
            self.ui.step1_import_pattern_label,
            self.ui.step1_study_name_lineedit,
            self.ui.step1_input_path_lineedit,
            self.ui.step1_import_format_combobox,
            self.ui.step1_import_raw_radio,
            self.ui.step1_import_epoched_radio,
            self.ui.step1_load_all_radio,
            self.ui.step1_load_pattern_radio,
            self.ui.step1_import_log_lineedit
        ]
        preprocessing_widgets = [
            self.ui.step2_montage_label,
            self.ui.step2_load_montage_radio,
            self.ui.step2_chanloc_path_lineedit,
            self.ui.step2_use_template_montage_radio,
            self.ui.step2_template_montage_combobox,
            self.ui.step2_filter_option_checkbox,
            self.ui.step2_downsamp_option_checkbox,
            self.ui.step2_auto_clean_option_checkbox,
            self.ui.step2_ch2rm_label,
            self.ui.step2_ch2rm_radio,
            self.ui.step2_ch2rm_combobox,
            self.ui.step2_prep_option_checkbox,
            self.ui.step2_save_path_button,
            self.ui.step2_save_path_lineedit,
            self.ui.step2_preprocess_data_button
        ]
        filter_sub_widgets = [
            self.ui.step2_filter_method_label,
            self.ui.step2_fir_filtermethod_radio,
            self.ui.step2_iir_filtermethod_radio,
            self.ui.step2_lowcut_freq_label,
            self.ui.step2_lowcut_freq_input,
            self.ui.step2_filt_hz1,
            self.ui.step2_highcut_freq_label,
            self.ui.step2_highcut_freq_input,
            self.ui.step2_filt_hz2
        ]
        downsample_sub_widgets = [
            self.ui.step2_downsamp_freq_label,
            self.ui.step2_downsamp_freq_input,
            self.ui.step2_downsamp_hz
        ]
        if self.ui.step1_import_data_radio.isChecked():
            widgets_to_rm = (
                    preprocessing_widgets +
                    filter_sub_widgets +
                    downsample_sub_widgets
            )
            set_widgets_status(import_data_widgets, mode='show')
            set_widgets_status(import_data_widgets, mode='enable')
            set_widgets_status(self.ui.step1_import_raw_button, mode='show')
            set_widgets_status(widgets_to_rm, mode='disable')
            set_widgets_status(widgets_to_rm, mode='hide')
            set_widgets_status(
                self.ui.step1_import_pattern_lineedit,
                'enable' if self.ui.step1_load_pattern_radio.isChecked()
                else 'disable')
            if (self.ui.step1_study_name_lineedit.text()
                    and self.ui.step1_input_path_lineedit):
                widgets_to_enable = [self.ui.step1_import_raw_button,
                                     self.ui.loaded_remove_file_button,
                                     self.ui.loaded_clear_files_button] + preprocessing_widgets
                set_widgets_status(widgets_to_enable, mode='enable')
            else:
                widgets_to_disable = [self.ui.loaded_remove_file_button,
                                      self.ui.loaded_clear_files_button] + plot_widgets + preprocessing_widgets
                set_widgets_status(widgets_to_disable, mode='disable')
        if self.ui.loaded_selected_files_list.count() != 0:
            set_widgets_status(self.ui.step2_preprocess_radio, mode='enable')
            if self.ui.loaded_selected_files_list.currentItem():
                set_widgets_status(plot_widgets, mode='enable')
                set_widgets_status(
                    self.ui.viz_plot_event_radio, 'enable' if
                    self.ui.step1_import_epoched_radio.isChecked() else 'disable'
                )

        else:
            set_widgets_status(self.ui.step2_preprocess_radio, mode='disable')
        if self.ui.step2_preprocess_radio.isChecked():
            widgets_to_show = preprocessing_widgets + filter_sub_widgets + downsample_sub_widgets
            widgets_to_hide_or_disable = [self.ui.step1_import_raw_button] + import_data_widgets
            set_widgets_status(widgets_to_show, mode='show')
            set_widgets_status(widgets_to_hide_or_disable, mode='disable')
            set_widgets_status(widgets_to_hide_or_disable, mode='hide')
            widget_conditions = [
                (self.ui.step2_template_montage_combobox, self.ui.step2_use_template_montage_radio.isChecked()),
                (self.ui.step2_chanloc_path_lineedit, self.ui.step2_load_montage_radio.isChecked()),
                (self.ui.step2_ch2rm_combobox, self.ui.step2_ch2rm_radio.isChecked()),
                (self.ui.step2_auto_clean_option_checkbox, self.ui.step1_import_raw_radio.isChecked()),
            ]
            for widget, condition in widget_conditions:
                set_widgets_status(widget, 'enable' if condition else 'disable')
            self.auto_clean_data = self.ui.step2_auto_clean_option_checkbox.isChecked()
            set_widgets_status(self.ui.step2_prep_option_checkbox, 'disable' if self.auto_clean_data else 'enable')
            if self.auto_clean_data:
                self.prep_data = self.ui.step2_prep_option_checkbox.isChecked()
            self.filter_data = self.ui.step2_filter_option_checkbox.isChecked()
            set_widgets_status(filter_sub_widgets, mode='enable' if self.filter_data else 'disable')
            if not self.filter_data:
                self.lowcut_freq = ''
                self.highcut_freq = ''
            self.downsample_data = self.ui.step2_downsamp_option_checkbox.isChecked()
            set_widgets_status(downsample_sub_widgets, mode='enable' if self.downsample_data else 'disable')
            set_widgets_status(self.ui.step2_preprocess_data_button,
                               'enable' if self.ui.step2_save_path_lineedit.text() else 'disable')

    def choose_input(self):
        """
        Open a file dialog to select the folder containing raw data
        """
        self.input_folder = QFileDialog.getExistingDirectory(self, "Select the folder containing raw data")
        self.ui.step1_input_path_lineedit.setText(self.input_folder)
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
                                             'csd', 'elp', 'bvef', 'csv', 'tsv', 'xyz', 'mat', 'ced']
                if chan_loc_extension not in valid_chan_loc_extensions:
                    QMessageBox.information(
                        self, "Load error",
                        "File extension is expected to be:"
                        "‘.loc’ or ‘.locs’ or ‘.eloc’ or '.ced (for EEGLAB files),"
                        "‘.sfp’ (BESA/EGI files), ‘.csd’, ‘.elc’, ‘.txt’, ‘.csd’, ‘.elp’ (BESA spherical),"
                        "‘.bvef’ (BrainVision files), ‘.csv’, ‘.tsv’, ‘.xyz’ (XYZ coordinates),"
                        "‘.mat’ (for Brainstorm files)'",
                        QMessageBox.Ok
                    )
                    self.comet_tbx.channel_location_dir = ''
                    return
                else:
                    self.comet_tbx.channel_location_dir = fname
                    self.ui.step2_chanloc_path_lineedit.setText(fname)

    def load_template_montage(self):
        """
        Set the channel location using the selected template montage.
        """
        self.comet_tbx.channel_location_dir = self.ui.step2_template_montage_combobox.currentText()

    def get_extension(self):
        """
        Get the selected extension from the data extension combo box.
        """
        selected_extension = self.ui.step1_import_format_combobox.currentText()
        extension = selected_extension.split('(')[1]
        extension = re.split(', | .', extension)[0]
        if extension.endswith(')'):
            extension = extension[:-1]
        return extension

    def get_data_type(self):
        """
        Get the selected data type based on the radio button.
        """
        return "epoched" if self.ui.step1_import_epoched_radio.isChecked() else "raw"

    def update_channel_names(self):
        """
        Update the channel names based on the selected EEG file.
        """
        filename = self.ui.loaded_selected_files_list.currentItem().text()
        eeg = DataIO().load_eeg(filename, self.comet_tbx.datatype, self.comet_tbx.channel_location_dir)
        data_channel_names = eeg.info['ch_names']
        self.ui.step2_ch2rm_combobox.addItems(data_channel_names)
        montage = None
        if not np.isnan(eeg.info['chs'][0]['loc'][0]):
            montage = eeg.get_montage()
        elif self.ui.step2_load_montage_radio.isChecked() and os.path.isfile(
                self.ui.step2_chanloc_path_lineedit.text()):
            montage = read_custom_montage(self.ui.step2_chanloc_path_lineedit.text())
        elif self.ui.step2_use_template_montage_radio.isChecked():
            if not self.comet_tbx.channel_location_dir:
                self.comet_tbx.channel_location_dir = "standard_1020"
            montage = make_standard_montage(self.comet_tbx.channel_location_dir)
        eeg.set_montage(montage, match_case=False, on_missing='warn')

    def load_raw(self):
        """
        Load raw EEG data and update the selected files list.
        """
        self.ui.loaded_selected_files_list.clear()
        self.comet_tbx.load_all_files = self.ui.step1_load_all_radio.isChecked()
        self.comet_tbx.pattern_content = self.ui.step1_import_pattern_lineedit.text()
        self.comet_tbx.input_folder = self.input_folder
        self.comet_tbx.extension = self.get_extension()
        self.comet_tbx.datatype = self.get_data_type()
        self.comet_tbx.load_raw()
        for i in range(len(self.comet_tbx.list_eegs_path)):
            self.ui.loaded_selected_files_list.addItem(str(self.comet_tbx.list_eegs_path[i]))
        self.ui.step1_import_log_lineedit.setText(
            f"{len(self.comet_tbx.list_eegs_path)} EEG data were detected."
        )
        if len(self.comet_tbx.list_eegs_path) > 0:
            self.ui.step2_preprocess_radio.setChecked(True)
        self.newstudy_controller()

    def new_study_save_path(self):
        """
        Choose and set the save directory for the new study.
        """
        save_parent_directory = QFileDialog.getExistingDirectory(
            self, "Select a parent folder to create a new study folder within.")
        self.study_name = self.ui.step1_study_name_lineedit.text()
        if not os.path.exists(save_parent_directory):
            print("Unable to find selected directory")
            return
        if not self.study_name:
            self.study_name = "EEG_COMET_NEW_STUDY"
            self.ui.step1_study_name_lineedit.setText(self.study_name)
        save_directory = os.path.join(save_parent_directory, self.study_name)
        if os.path.exists(save_directory):
            QMessageBox.information(self, "A folder with the same study name already exists!",
                                    "Please choose another directory or rename your study.",
                                    QMessageBox.Ok)
        else:
            self.save_dir = save_directory
            self.ui.step2_save_path_lineedit.setText(self.save_dir)
        self.newstudy_controller()

    def remove_file(self):
        """
        Remove the selected files from the selected files list.
        """
        selected_items = self.loaded_selected_files_list.selectedItems()
        if not selected_items:
            return
        [self.loaded_selected_files_list.takeItem(self.loaded_selected_files_list.row(item)) for item in selected_items]
        self.newstudy_controller()

    def clear_files(self):
        """
        Clear all files from the selected files list.
        """
        self.ui.loaded_selected_files_list.clear()
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
        self.filter_data = self.ui.step2_filter_option_checkbox.isChecked()
        if self.filter_data:
            if self.ui.step2_lowcut_freq_input.text() >= self.ui.step2_highcut_freq_input.text():
                QMessageBox.information(self, "Filter Error",
                                        "Please modify the filter range!",
                                        QMessageBox.Ok)
            self.filter_method = 'fir' if self.ui.step2_fir_filtermethod_radio.isChecked() else 'iir'
            self.lowcut_freq = int(self.ui.step2_lowcut_freq_input.text())
            self.highcut_freq = int(self.ui.step2_highcut_freq_input.text())
        else:
            self.filter_method = ''
            self.lowcut_freq = ''
            self.highcut_freq = ''
        self.downsample_data = self.ui.step2_downsamp_option_checkbox.isChecked()
        self.sample_rate = int(self.ui.step2_downsamp_freq_input.text()) if self.downsample_data else ''
        self.auto_clean_data = self.ui.step2_auto_clean_option_checkbox.isChecked()
        self.chan2rm = (self.ui.step2_ch2rm_combobox.currentData() if self.ui.step2_ch2rm_radio.isChecked()
                        else 'missing')
        self.prep_data = self.ui.step2_prep_option_checkbox.isChecked()
        attributes = ['preprocessed_data_path', 'filter_data', 'filter_method', 'lowcut_freq', 'highcut_freq',
                      'downsample_data', 'sample_rate', 'chan2rm', 'auto_clean_data', 'prep_data', 'save_dir']
        for attr in attributes:
            setattr(self.comet_tbx, attr, getattr(self, attr))
        self.comet_tbx.study_name = self.ui.step1_study_name_lineedit.text()
        self.comet_tbx.eeg_info_path = os.path.join(self.comet_tbx.save_dir, "eeg_info.pkl")
        self.comet_tbx.do_preprocessing()
        self.done_preprocessing = True
        self.comet_tbx.save_tbx()
        if self.main_window:
            self.main_window.load_study(from_new_study=True)
        self.ui.close()

    def item_selected(self):
        """
        Retrieve the full file path and name from the selected item.
        """
        filepath = self.ui.loaded_selected_files_list.currentItem().text()
        filename = os.path.splitext(os.path.basename(filepath))[0]
        self.ui.vis_figure_filename_lineedit.setText(filename)
        return filepath, filename

    def plot_montage(self):
        """
        Plot EEG montage and display channel names if selected.
        """
        self.init_canvas()
        filepath, filename = self.item_selected()
        eeg = DataIO().load_eeg(filepath, self.comet_tbx.datatype, self.comet_tbx.channel_location_dir, preload=False)
        if eeg.info['dig'] is None:
            QMessageBox.information(self, "Load error",
                                    "Unable to retrieve channel locations."
                                    "Please ensure they are imported before proceeding.",
                                    QMessageBox.Ok)
        else:
            show_names = self.ui.vis_channel_names_checkbox.isChecked()
            if self.ui.step2_ch2rm_combobox.currentData():
                self.chan2rm = self.ui.step2_ch2rm_combobox.currentData()
                eeg.info["bads"].extend(self.chan2rm)
            fig, _ = eeg.plot_sensors(kind='select', show_names=show_names, show=False)
            self.ui.vis_figure_title_lineedit.setText("EEG Montage")
            self.canvas.figure = fig
            self.canvas.draw()

    def plot_eeg(self):
        """
        Plot the EEG data using the PyQt application canvas.
        """
        filepath, filename = self.item_selected()
        eeg = DataIO().load_eeg(filepath, self.comet_tbx.datatype, self.comet_tbx.channel_location_dir)

        if self.ui.viz_plot_time_radio.isChecked():
            self.clean_figure_layout()
            fig = eeg.plot(verbose='ERROR')
            fig.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            self.ui.Figure_Layout.addWidget(fig)

        if self.ui.viz_plot_event_radio.isChecked():
            self.init_canvas()
            tmin = -0.2
            tmax = 0.4
            time_points = [30, 45, 60, 100, 180, 280]
            time_points_sec = np.array(time_points) / 1000.0
            times = eeg.times
            tmin_idx = np.searchsorted(times, tmin)
            tmax_idx = np.searchsorted(times, tmax)
            epoched_data = eeg.get_data(copy=True)
            avg_data = epoched_data.mean(axis=0)
            fig = self.canvas.figure
            gs = fig.add_gridspec(2, len(time_points), height_ratios=[1, 2])
            for i, time_point in enumerate(time_points_sec):
                ax_topo = fig.add_subplot(gs[0, i])
                plot_topomap(avg_data[:, np.searchsorted(times, time_point)], eeg.info,
                             axes=ax_topo, show=False)
                ax_topo.set_title(f'{time_point * 1000:.0f} ms', fontsize=16)
            ax_main = fig.add_subplot(gs[1, :])
            for i, channel_data in enumerate(avg_data):
                ax_main.plot(times[tmin_idx:tmax_idx] * 1000, channel_data[tmin_idx:tmax_idx] * 1e6)
            ax_main.axvline(0, color='k', linestyle='--', label='Event Onset')
            xticks = np.arange(int(tmin * 1000), int(tmax * 1000) + 1, 50)
            xticks = np.unique(np.concatenate((xticks, time_points)))
            ax_main.set_xticks(xticks)
            ax_main.set_xticklabels([f'{int(x)}' for x in xticks])
            for time_point in time_points:
                ax_main.axvline(time_point, color='r', linestyle='--', alpha=0.7)
            ax_main.set_xlabel('Time (ms)', fontsize=18)
            ax_main.set_ylabel('Amplitude (μV)', fontsize=18)
            ax_main.set_title(f'{filename}', fontsize=20)
            ax_main.tick_params(axis='both', which='major', labelsize=16)
            self.canvas.draw()
            self.ui.vis_figure_title_lineedit.setText("Butterfly plot of TMS‐evoked potentials")

    def plot_psd(self):
        """
        Plot the Power Spectral Density (PSD) of EEG data.
        """
        self.init_canvas()
        filepath, filename = self.item_selected()
        eeg = DataIO().load_eeg(filepath, self.comet_tbx.datatype, self.comet_tbx.channel_location_dir, preload=False)
        fmin_plot = int(self.ui.vis_range_psd_min.text())
        fmax_plot = int(self.ui.vis_range_psd_max.text())
        fig = eeg.compute_psd(fmin=fmin_plot, fmax=fmax_plot, verbose='ERROR').plot(show=False)
        self.canvas.figure = fig
        self.ui.vis_figure_title_lineedit.setText("Power Spectral Density (PSD) using Multitapers")
        self.canvas.draw()
