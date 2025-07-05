import os.path
import re
import numpy as np
from mne.channels import get_builtin_montages, read_custom_montage, make_standard_montage
from mne.viz import plot_topomap
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from gui_utils.CheckableComboBox import CheckableComboBox
from gui_utils.set_widgets_status import set_widgets_status
from data_utils.data_io import DataIO


class NewStudyWindow(QDialog):
    def __init__(self, context, parent=None, main_window=None, comet_tbx=None):
        super().__init__(parent)
        self.main_window = main_window
        self.comet = comet_tbx
        self.ui = uic.loadUi(context.get_resource("NewStudyWindow.ui"), self)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.ui.setWindowTitle("New Study - Import EEG Data and Preprocess")
        self.init_ui_components()
        self.setup_connections()
        self.comet.montage = ""
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
        # Group connections by target function

        # 1. Items that trigger newstudy_controller
        controller_widgets = {
            # Radio buttons and checkboxes (clicked signal)
            'clicked': [
                self.ui.step1_import_epoched_radio,
                self.ui.step1_import_raw_radio,
                self.ui.step1_load_all_radio,
                self.ui.step1_load_pattern_radio,
                self.ui.step2_load_montage_radio,
                self.ui.step2_use_template_montage_radio,
                self.ui.step2_temporal_filter_option_checkbox,
                self.ui.step2_downsamp_option_checkbox,
                self.ui.step2_spatial_filter_option_checkbox,
                self.ui.step2_ch2rm_radio
            ],
            # Text inputs (textChanged signal)
            'textChanged': [
                self.ui.step1_study_name_lineedit,
                self.ui.step1_import_pattern_lineedit
            ],
            # List widgets (itemClicked signal)
            'itemClicked': [
                self.ui.loaded_selected_files_list
            ]
        }

        # 2. Items that trigger plot_montage
        montage_widgets = {
            'clicked': [
                self.ui.vis_montage_button,
            ],
            'itemClicked': [
                self.ui.loaded_selected_files_list
            ],
            'itemCheckedStateChanged': [
                self.ui.step2_ch2rm_combobox
            ],
            'textChanged': [
                self.ui.step2_chanloc_path_lineedit
            ],
            'currentTextChanged': [
                self.ui.step2_template_montage_combobox
            ]
        }

        # 3. Specific button actions
        action_widgets = [
            (self.ui.step1_input_path_button, 'clicked', self.choose_input),
            (self.ui.step1_import_raw_button, 'clicked', self.load_raw),
            (self.ui.step2_load_montage_radio, 'clicked', self.load_custom_montage),
            (self.ui.step2_template_montage_combobox, 'activated', self.load_template_montage),
            (self.ui.step2_save_path_button, 'clicked', self.new_study_save_path),
            (self.ui.step2_preprocess_data_button, 'clicked', self.preprocess_data),
            (self.ui.loaded_remove_file_button, 'clicked', self.remove_file),
            (self.ui.loaded_clear_files_button, 'clicked', self.clear_files),
            (self.ui.vis_psd_button, 'clicked', self.plot_psd),
            (self.ui.vis_plot_button, 'clicked', self.plot_eeg),
        ]

        # Connect widgets that trigger newstudy_controller
        for signal_name, widgets in controller_widgets.items():
            for widget in widgets:
                getattr(widget, signal_name).connect(self.newstudy_controller)

        # Connect widgets that trigger plot_montage
        for signal_name, widgets in montage_widgets.items():
            for widget in widgets:
                getattr(widget, signal_name).connect(self.plot_montage)

        # Connect special cases: update_channel_names
        self.ui.loaded_selected_files_list.itemClicked.connect(self.update_channel_names)

        # Connect specific actions
        for widget, signal_name, action in action_widgets:
            getattr(widget, signal_name).connect(action)

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
            self.ui.vis_montage_button,
            self.ui.vis_psd_button
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
            self.ui.step2_temporal_filter_option_checkbox,
            self.ui.step2_downsamp_option_checkbox,
            self.ui.step2_spatial_filter_option_checkbox,
            self.ui.step2_preprocess_label,
            self.ui.step2_ch2rm_label,
            self.ui.step2_ch2rm_radio,
            self.ui.step2_ch2rm_combobox,
            self.ui.step2_prep_option_checkbox,
            self.ui.step2_save_path_button,
            self.ui.step2_save_path_lineedit,
            self.ui.step2_preprocess_data_button
        ]

        temporal_filter_sub_widgets = [
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

        if self.ui.new_study_tab_widget.currentIndex() == 0:
            widgets_to_rm = (
                    preprocessing_widgets +
                    temporal_filter_sub_widgets +
                    downsample_sub_widgets
            )
            set_widgets_status(import_data_widgets, mode='show')
            set_widgets_status(import_data_widgets, mode='enable')
            set_widgets_status(self.ui.step1_import_raw_button, mode='show')
            set_widgets_status(widgets_to_rm, mode='disable')
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
            self.ui.new_study_tab_widget.setTabEnabled(1, True)
            if self.ui.loaded_selected_files_list.currentItem():
                set_widgets_status(plot_widgets, mode='enable')
        else:
            self.ui.new_study_tab_widget.setTabEnabled(1, False)

        if self.ui.new_study_tab_widget.currentIndex() == 1:
            widgets_to_show = preprocessing_widgets + temporal_filter_sub_widgets + downsample_sub_widgets
            widgets_to_hide_or_disable = [self.ui.step1_import_raw_button] + import_data_widgets
            set_widgets_status(widgets_to_show, mode='show')
            set_widgets_status(widgets_to_hide_or_disable, mode='disable')
            widget_conditions = [
                (self.ui.step2_template_montage_combobox, self.ui.step2_use_template_montage_radio.isChecked()),
                (self.ui.step2_chanloc_path_lineedit, self.ui.step2_load_montage_radio.isChecked()),
                (self.ui.step2_ch2rm_combobox, self.ui.step2_ch2rm_radio.isChecked())
            ]
            for widget, condition in widget_conditions:
                set_widgets_status(widget, 'enable' if condition else 'disable')
            self.temporal_filter_data = self.ui.step2_temporal_filter_option_checkbox.isChecked()
            set_widgets_status(temporal_filter_sub_widgets, mode='enable' if self.temporal_filter_data else 'disable')
            if not self.temporal_filter_data:
                self.lowcut_freq = ''
                self.highcut_freq = ''
            self.downsample_data = self.ui.step2_downsamp_option_checkbox.isChecked()
            set_widgets_status(downsample_sub_widgets, mode='enable' if self.downsample_data else 'disable')
            self.spatial_filter_data = self.ui.step2_spatial_filter_option_checkbox.isChecked()
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
                        "'.loc' or '.locs' or '.eloc' or '.ced (for EEGLAB files),"
                        "'.sfp' (BESA/EGI files), '.csd', '.elc', '.txt', '.csd', '.elp' (BESA spherical),"
                        "'.bvef' (BrainVision files), '.csv', '.tsv', '.xyz' (XYZ coordinates),"
                        "'.mat' (for Brainstorm files)'",
                        QMessageBox.Ok
                    )
                    self.comet.montage = ''
                    return
                else:
                    self.comet.montage = fname
                    self.ui.step2_chanloc_path_lineedit.setText(fname)

    def load_template_montage(self):
        """
        Set the channel location using the selected template montage.
        """
        self.comet.montage = self.ui.step2_template_montage_combobox.currentText()

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
        eeg = DataIO().load_eeg(filename, self.comet.datatype, self.comet.montage)
        data_channel_names = eeg.info['ch_names']
        self.ui.step2_ch2rm_combobox.addItems(data_channel_names)
        montage = None
        if not np.isnan(eeg.info['chs'][0]['loc'][0]):
            montage = eeg.get_montage()
        elif self.ui.step2_load_montage_radio.isChecked() and os.path.isfile(
                self.ui.step2_chanloc_path_lineedit.text()):
            montage = read_custom_montage(self.ui.step2_chanloc_path_lineedit.text())
        elif self.ui.step2_use_template_montage_radio.isChecked():
            if not self.comet.montage:
                self.comet.montage = "standard_1020"
            montage = make_standard_montage(self.comet.montage)
        eeg.set_montage(montage, match_case=False, on_missing='warn')

    def load_raw(self):
        """
        Load raw EEG data and update the selected files list.
        """
        self.ui.loaded_selected_files_list.clear()
        self.comet.load_all_files = self.ui.step1_load_all_radio.isChecked()
        self.comet.pattern_content = self.ui.step1_import_pattern_lineedit.text()
        self.comet.input_folder = self.input_folder
        self.comet.extension = self.get_extension()
        self.comet.datatype = self.get_data_type()
        self.comet.load_raw()
        for i in range(len(self.comet.list_eegs_path)):
            self.ui.loaded_selected_files_list.addItem(str(self.comet.list_eegs_path[i]))
        self.ui.step1_import_log_lineedit.setText(
            f"{len(self.comet.list_eegs_path)} EEG data were detected."
        )
        if len(self.comet.list_eegs_path) > 0:
            self.ui.new_study_tab_widget.setCurrentIndex(1)
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
        # Set critical path parameters first
        self.study_name = self.ui.step1_study_name_lineedit.text()
        save_parent_dir = os.path.dirname(self.ui.step2_save_path_lineedit.text())

        # Update COMET critical path parameters first
        self.comet.study_name = self.study_name
        self.comet.input_folder = self.ui.step1_input_path_lineedit.text()
        self.comet.output_folder = save_parent_dir

        # Set preprocessing parameters
        self.temporal_filter_data = self.ui.step2_temporal_filter_option_checkbox.isChecked()
        if self.temporal_filter_data:
            if self.ui.step2_lowcut_freq_input.text() >= self.ui.step2_highcut_freq_input.text():
                QMessageBox.information(self, "Temporal Filter Error",
                                        "Please modify the filter range!",
                                        QMessageBox.Ok)
                return
            self.filter_method = 'fir' if self.ui.step2_fir_filtermethod_radio.isChecked() else 'iir'
            self.lowcut_freq = int(self.ui.step2_lowcut_freq_input.text())
            self.highcut_freq = int(self.ui.step2_highcut_freq_input.text())
        else:
            self.filter_method = ''
            self.lowcut_freq = ''
            self.highcut_freq = ''

        self.downsample_data = self.ui.step2_downsamp_option_checkbox.isChecked()
        self.sample_rate = int(self.ui.step2_downsamp_freq_input.text()) if self.downsample_data else ''
        self.spatial_filter_data = self.ui.step2_spatial_filter_option_checkbox.isChecked()
        self.chan2rm = (self.ui.step2_ch2rm_combobox.currentData() if self.ui.step2_ch2rm_radio.isChecked()
                        else 'missing')
        self.prep_data = self.ui.step2_prep_option_checkbox.isChecked()

        # The save_dir should match the one that COMET will compute based on study_name and output_folder
        self.save_dir = os.path.join(save_parent_dir, self.study_name)

        # Set all processing attributes in COMET
        attributes = [
            'temporal_filter_data', 'filter_method', 'lowcut_freq', 'highcut_freq',
            'downsample_data', 'sample_rate', 'spatial_filter_data', 'chan2rm', 'prep_data'
        ]
        for attr in attributes:
            setattr(self.comet, attr, getattr(self, attr))

        # Initialize log window and log study creation
        if not hasattr(self.comet, 'LogWindow') or self.comet.LogWindow is None:
            self.comet.initialize_log_window()
        
        self.comet.LogWindow.append_log("Study Creation", log_type='section')
        study_creation_info = (
            f"Study Name: {self.study_name}\n"
            f"Input Directory: {self.comet.input_folder}\n"
            f"Output Directory: {self.save_dir}\n"
            f"Data Type: {self.comet.datatype}\n"
            f"File Extension: {self.comet.extension}"
        )
        self.comet.LogWindow.append_log(study_creation_info, log_type='info')
        self.comet.LogWindow.append_log(f"New study '{self.study_name}' has been created successfully", log_type='success')

        # Now let COMET handle directory creation and processing
        self.comet.run_preprocessing()

        if self.main_window:
            self.main_window.load_study(from_new_study=True)
            self.main_window.main_tab.setCurrentIndex(0)
        self.close()

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

        try:
            eeg = DataIO().load_eeg(filepath, self.comet.datatype, self.comet.montage,
                                    preload=False)

            # Check if digital points exist
            if eeg.info['dig'] is None:
                QMessageBox.information(self, "Load error",
                                        "Unable to retrieve channel locations. "
                                        "Please ensure they are imported before proceeding.",
                                        QMessageBox.Ok)
                return

            # Check if channels actually have positions
            if not any(ch.get('loc', None) is not None for ch in eeg.info['chs']):
                QMessageBox.information(self, "Position error",
                                        "Channel information exists but no valid positions found. "
                                        "Please check your montage file.",
                                        QMessageBox.Ok)
                return

            # Proceed with plotting if we have channel positions
            if self.ui.step2_ch2rm_combobox.currentData():
                self.chan2rm = self.ui.step2_ch2rm_combobox.currentData()
                eeg.info["bads"].extend(self.chan2rm)

            try:
                fig, _ = eeg.plot_sensors(kind='select', show_names=True, show=False)
                self.ui.vis_figure_title_lineedit.setText("EEG Montage")
                self.canvas.figure = fig
                self.canvas.draw()
            except RuntimeError as e:
                error_msg = str(e)
                QMessageBox.warning(self, "Plotting Error",
                                    f"Could not plot montage: {error_msg}\n"
                                    "Please ensure your data has valid channel positions.",
                                    QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"An unexpected error occurred: {str(e)}",
                                 QMessageBox.Ok)

    def plot_eeg(self):
        """
        Plot the EEG data using the PyQt application canvas.
        Automatically detects whether data is epoched (evoked) or raw.
        """
        filepath, filename = self.item_selected()
        eeg = DataIO().load_eeg(filepath, self.comet.datatype, self.comet.montage)

        # Automatically determine plot type based on data type
        if self.comet.datatype == "epoched":
            # Plot evoked/epoched data with topomaps
            self.init_canvas()
            tmin = -0.2
            tmax = 0.4
            time_points = [50, 100, 150, 200, 250, 300, 350]
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
                ax_main.plot(times[tmin_idx:tmax_idx] * 1000,
                             channel_data[tmin_idx:tmax_idx] * 1e6)
            ax_main.axvline(0, color='k', linestyle='--', label='Event Onset')
            for tp in time_points:
                ax_main.axvline(tp, color='r', linestyle='--', alpha=0.7)
            ax_main.set_axisbelow(True)  # draw grid behind the lines
            ax_main.grid(True, axis='y', which='major', linestyle='--', alpha=0.5)
            xticks = np.arange(int(tmin * 1000), int(tmax * 1000) + 1, 50)
            xticks = np.unique(np.concatenate((xticks, time_points)))
            ax_main.set_xticks(xticks)
            ax_main.set_xticklabels([f'{int(x)}' for x in xticks])
            ax_main.set_xlabel('Time (ms)', fontsize=18)
            ax_main.set_ylabel('Amplitude (μV)', fontsize=18)
            ax_main.set_title(f'{filename} - Evoked Response', fontsize=20)
            ax_main.tick_params(axis='both', which='major', labelsize=16)
            self.canvas.draw()
        else:
            # Plot raw continuous data
            self.clean_figure_layout()
            fig = eeg.plot(
                n_channels=min(15, len(eeg.ch_names)),
                duration=5.0,
                scalings="auto",
                show=False,
                block=False,
                title=f"{filename} - Raw EEG",
                overview_mode="hidden",
                verbose="ERROR"
            )
            self.ui.Figure_Layout.addWidget(fig)
            self.ui.vis_figure_title_lineedit.setText("Raw EEG Time Series")

    def plot_psd(self):
        """
        Plot the Power Spectral Density (PSD) of EEG data.
        """
        self.init_canvas()
        filepath, filename = self.item_selected()
        eeg = DataIO().load_eeg(filepath, self.comet.datatype, self.comet.montage, preload=False)
        fig = eeg.compute_psd(fmin=1, fmax=50, verbose='ERROR').plot(show=False)
        self.canvas.figure = fig
        self.ui.vis_figure_title_lineedit.setText("Power Spectral Density (PSD)")
        self.canvas.draw()
