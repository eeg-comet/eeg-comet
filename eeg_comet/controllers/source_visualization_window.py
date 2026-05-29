"""3D source visualization window for EEG-COMET localized sources.

Provides interactive and static visualizations of localized cortical sources
for microstate analyses, including subject/time selection, averaged views,
and export utilities.
"""

import contextlib
import datetime
import os.path
import traceback

import mne
import numpy as np
import pyvista as pv
from PyQt5 import uic
from PyQt5.QtWidgets import QAbstractItemView, QDialog
from pyvistaqt import QtInteractor

from eeg_comet.gui_utils.responsive import (
    apply_window_minimum,
    configure_splitter,
    expand_canvas,
)
from eeg_comet.sourcelocalization_utils.source_io import SourceIO
from eeg_comet.sourcelocalization_utils.source_visualizer import SourceVisualizer

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SourceVisualizationWindow(QDialog):
    """Dialog for 3D source visualization and interaction.

    Attributes:
      list_subjects (list[str] | None): Subjects discovered under the STC folder.
      comet: COMET toolbox instance providing paths, labels, and settings.
      context: Resource provider used to load the Qt UI.
      subjects_dir (str): FreeSurfer subjects directory.
      source_visualizer (SourceVisualizer): Mesh/visualization helper.
      source_io (SourceIO): STC discovery and IO helper.
      plotter (QtInteractor): Embedded PyVista 3D plotter widget.
    """

    def __init__(self, context, parent=None, comet_tbx=None):
        """Initialize the source visualization window and 3D plotter.

        Args:
          context: Resource/context provider to resolve UI assets.
          parent: Optional parent widget.
          comet_tbx: COMET toolbox instance.
        """
        super().__init__(parent)

        self.list_subjects = None
        self.comet = comet_tbx
        self.context = context

        # Set up subjects directory
        if self.comet.use_anatomy != "fsaverage":
            self.subjects_dir = self.comet.individual_subjects_dir
        else:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            self.subjects_dir = os.path.dirname(fs_dir)

        # Initialize SourceVisualizer and SourceIO
        self.source_visualizer = SourceVisualizer(
            self.comet.anatomy_subjects_dir, self.comet.spacing, self.comet.localized_sources_path
        )

        # Initialize SourceIO
        self.source_io = SourceIO()

        self.ui = uic.loadUi(context.get_resource("SourceVisualizationWindow.ui"), self)
        self.ui.setWindowTitle("Visualization of the localized sources")
        apply_window_minimum(self, "tool")
        if hasattr(self.ui, "source_visualization_label"):
            self.ui.source_visualization_label.setProperty("role", "banner")
        if hasattr(self.ui, "splitter"):
            configure_splitter(
                self.ui.splitter,
                ratio=(1, 2),
                save_key="source_visualization",
            )

        for m in self.comet.micro_labels:
            self.ui.microstate_combobox.addItem(m)

        self.plotter = QtInteractor(self)
        self.plotter.enable_image_style()
        expand_canvas(self.plotter, minimum=(360, 280))
        self.Figure_Layout.addWidget(self.plotter)

        self.locate_subjects_dir()
        self.setup_connections()
        self.resize(1000, 800)

    def setup_connections(self):
        """Set up all UI connections."""
        self.ui.stc_time_slider.valueChanged.connect(self.source_localization_controller)
        self.ui.show_stc_button.clicked.connect(self.show_sources)
        self.ui.show_microstate_sources_button.clicked.connect(self.show_microstate_sources)
        self.ui.subjects_list.itemSelectionChanged.connect(self.handle_new_file_selection)

        # Connect the show all microstate sources button
        if hasattr(self.ui, "show_all_microstate_sources_button"):
            self.ui.show_all_microstate_sources_button.clicked.connect(
                self.show_all_microstate_sources
            )

    def source_localization_controller(self):
        """Update time input when the slider value changes."""
        self.ui.stc_time_input.setText(str(self.ui.stc_time_slider.value()))

    def locate_subjects_dir(self):
        """Populate the subjects list from the STC directory."""
        self.ui.subjects_list.clear()
        subjects_dir = os.path.join(self.comet.localized_sources_path, "stc")

        if os.path.exists(subjects_dir):
            self.list_subjects = [
                folder
                for folder in os.listdir(subjects_dir)
                if os.path.isdir(os.path.join(subjects_dir, folder))
            ]
            for list_subject in self.list_subjects:
                self.ui.subjects_list.addItem(str(list_subject))
        else:
            self.list_subjects = []

        self.ui.subjects_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def handle_new_file_selection(self):
        """Handle selection changes in the subjects list."""
        self.ui.microstates_list_combobox.clear()
        self.plotter.clear()

        selected_items = self.ui.subjects_list.selectedItems()

        if selected_items:
            subject_name = selected_items[0].text()
            self.setup_time_range_for_subject(subject_name)

            self.selected_subjects = [item.text() for item in selected_items]

            # Check TESS sources
            self.tess_dir = os.path.join(self.comet.localized_sources_path, "tess_sources")
            list_tess_subjects = []
            if os.path.exists(self.tess_dir):
                list_tess_subjects = [
                    folder
                    for folder in self.selected_subjects
                    if os.path.isdir(os.path.join(self.tess_dir, folder))
                ]

            # Check averaged sources
            self.avg_dir = os.path.join(self.comet.localized_sources_path, "avg_sources")
            list_avg_subjects = []
            if os.path.exists(self.avg_dir):
                list_avg_subjects = [
                    folder
                    for folder in self.selected_subjects
                    if os.path.isdir(os.path.join(self.avg_dir, folder))
                    and self._subject_has_microstate_files(folder)
                ]

            plottable = False

            # Add TESS options if available
            if len(list_tess_subjects) == len(self.selected_subjects):
                self.ui.microstates_list_combobox.addItems(
                    ["TESS - Filtered Z-Scores", "TESS - Raw Z-Scores"]
                )
                plottable = True

            # Add AVG options if available
            if len(list_avg_subjects) == len(self.selected_subjects):
                self.ui.microstates_list_combobox.addItems(
                    ["AVG - Single Source", "AVG - All Sources"]
                )
                plottable = True

            if not plottable:
                self.ui.microstates_list_combobox.addItems(["--Data Unavailable--"])

    def _find_all_subjects_with_avg_data(self):
        """Find all subjects that have complete averaged source data.

        Returns:
          list[str]: Subject IDs with complete averaged microstate files.
        """
        avg_base_path = os.path.join(self.comet.localized_sources_path, "avg_sources")

        if not os.path.exists(avg_base_path):
            print(f"Averaged sources directory does not exist: {avg_base_path}")
            return []

        # Get all subject directories
        all_subject_dirs = [
            d for d in os.listdir(avg_base_path) if os.path.isdir(os.path.join(avg_base_path, d))
        ]

        print(f"Found {len(all_subject_dirs)} subject directories: {all_subject_dirs}")

        # Filter subjects that have all required microstate files
        valid_subjects = []
        for subject_name in all_subject_dirs:
            if self._subject_has_microstate_files(subject_name):
                valid_subjects.append(subject_name)
                print(f"  ✓ {subject_name}: Complete microstate data")
            else:
                print(f"  ✗ {subject_name}: Missing microstate files")

        return valid_subjects

    def _subject_has_microstate_files(self, subject_name):
        """Check if a subject has all required microstate files.

        Args:
          subject_name (str): Subject identifier.

        Returns:
          bool: True if all expected files exist.
        """
        avg_base_path = os.path.join(self.comet.localized_sources_path, "avg_sources")
        subject_path = os.path.join(avg_base_path, subject_name)

        if not os.path.exists(subject_path):
            return False

        files = [f for f in os.listdir(subject_path) if f.endswith(".npy")]
        expected_files = [f"{subject_name}_{label}.npy" for label in self.comet.micro_labels]

        # Check if all expected files exist
        missing_files = [f for f in expected_files if f not in files]
        if missing_files:
            print(f"    Missing files for {subject_name}: {missing_files}")
            return False

        return True

    def setup_time_range_for_subject(self, subject_name):
        """Set the time slider range using STC data for a subject.

        Args:
          subject_name (str): Subject identifier used to locate STC files.
        """
        try:
            stc_files_dict = self.source_io.find_stc_files(
                self.comet.localized_sources_path, subject_name
            )

            if subject_name in stc_files_dict and stc_files_dict[subject_name]:
                stc_file_path = stc_files_dict[subject_name][0]
                stc_info = self.source_io.get_stc_info(stc_file_path)

                if stc_info:
                    min_time_ms = int(stc_info["tmin"] * 1000)
                    max_time_ms = int(stc_info["tmax"] * 1000)

                    self.ui.stc_time_slider.setMinimum(min_time_ms)
                    self.ui.stc_time_slider.setMaximum(max_time_ms)
                    self.ui.stc_time_slider.setValue(min_time_ms)
                    self.ui.stc_time_input.setText(str(min_time_ms))

        except Exception:
            pass

    def show_sources(self):
        """Load STC data and show sources at the specified time using SourceIO."""
        try:
            selected_items = self.ui.subjects_list.selectedItems()
            if not selected_items:
                return

            subject_name = selected_items[0].text()
            time_point = float(self.ui.stc_time_input.text()) / 1000.0

            stc_files_dict = self.source_io.find_stc_files(
                self.comet.localized_sources_path, subject_name
            )

            if subject_name not in stc_files_dict or not stc_files_dict[subject_name]:
                return

            stc_file_path = stc_files_dict[subject_name][0]
            stc = self.source_io.read_single_stc(stc_file_path)

            if stc is None:
                return

            if time_point < stc.tmin or time_point > stc.times[-1]:
                time_point = max(stc.tmin, min(stc.times[-1], time_point))

            self.plotter.clear()

            brain = stc.plot(
                subjects_dir=self.subjects_dir,
                initial_time=time_point,
                views=["lateral", "medial"],
                cortex="low_contrast",
                hemi="split",
                surface="inflated",
                time_viewer=False,
                background="white",
                size=(800, 800),
                colormap="hot",
                smoothing_steps=10,
                colorbar=True,
            )

            brain.add_text(
                0.1,
                0.9,
                f"Subject: {subject_name} | Time: {time_point:.3f}s",
                "title",
                font_size=14,
            )

            img = brain.screenshot()
            brain.close()

            plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10)
            tex = pv.numpy_to_texture(img)
            self.plotter.add_mesh(plane, texture=tex, name="brain_stc")

            self.plotter.view_xy()
            self.plotter.camera.zoom(1.2)
            self.plotter.update()

        except Exception:
            pass

    def show_microstate_sources(self):
        """Show microstate-specific sources averaged across selected subjects."""
        try:
            selected_items = self.ui.subjects_list.selectedItems()
            if not selected_items:
                return

            source_mode_text = self.ui.microstates_list_combobox.currentText()

            if "--Data Unavailable--" in source_mode_text:
                return

            if "AVG - All Sources" in source_mode_text:
                self.show_all_microstate_sources()
                return

            microstate_idx = self.ui.microstate_combobox.currentIndex()

            # Map the UI text to the expected source_mode values
            if "TESS - Filtered Z-Scores" in source_mode_text:
                result_type = "filtered"
                data_type = "tess"
            elif "TESS - Raw Z-Scores" in source_mode_text:
                result_type = "raw"
                data_type = "tess"
            elif "AVG - Single Source" in source_mode_text:
                data_type = "avg"
            else:
                return

            self.plotter.clear()
            selected_subjects = [item.text() for item in selected_items]

            if data_type == "tess":
                microstate_data = self._load_tess_data(
                    selected_subjects, microstate_idx, result_type
                )
            elif data_type == "avg":
                microstate_data = self._load_avg_data(selected_subjects, microstate_idx)

            if microstate_data is None:
                return

            # Normalize the data
            data_min = np.min(microstate_data)
            data_max = np.max(microstate_data)

            if data_max > data_min:
                microstate_data_norm = (microstate_data - data_min) / (data_max - data_min)
            else:
                microstate_data_norm = np.zeros_like(microstate_data)

            # Create MNE SourceEstimate object
            n_vertices_total = len(microstate_data_norm)
            n_vertices_per_hemi = n_vertices_total // 2

            vertices = [
                np.arange(n_vertices_per_hemi),  # Left hemisphere
                np.arange(n_vertices_per_hemi),  # Right hemisphere
            ]

            stc_data = microstate_data_norm[:, np.newaxis]

            mystc = mne.SourceEstimate(
                stc_data, vertices=vertices, tmin=0, tstep=1, subject="fsaverage"
            )

            # Plot the brain
            brain = mystc.plot(
                subjects_dir=self.subjects_dir,
                initial_time=0,
                views=["lateral", "medial"],
                cortex="low_contrast",
                hemi="split",
                surface="inflated",
                time_viewer=False,
                background="white",
                size=(800, 800),
                colormap="hot",
                smoothing_steps=10,
                colorbar=True,
                clim=dict(kind="percent", lims=[70, 85, 99]),
            )

            # Add title
            microstate_label_display = (
                self.comet.micro_labels[microstate_idx]
                if microstate_idx < len(self.comet.micro_labels)
                else f"State {microstate_idx}"
            )

            title_text = f"Microstate {microstate_label_display} - {source_mode_text}\nAveraged across {len(selected_subjects)} subjects"
            brain.add_text(0.1, 0.9, title_text, "title", font_size=12)

            img = brain.screenshot()
            brain.close()

            # Display in PyVista plotter
            plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10)
            tex = pv.numpy_to_texture(img)
            self.plotter.add_mesh(plane, texture=tex, name="brain_microstate")

            self.plotter.view_xy()
            self.plotter.camera.zoom(1.2)
            self.plotter.update()

        except Exception:
            pass

    def show_all_microstate_sources(self):
        """Show all microstate sources averaged across all available subjects."""
        try:
            self.plotter.clear()

            # Automatically find all subjects with averaged source data
            all_available_subjects = self._find_all_subjects_with_avg_data()

            if not all_available_subjects:
                print("No subjects found with averaged source data")
                return

            print(
                f"Found {len(all_available_subjects)} subjects with averaged source data: {all_available_subjects}"
            )

            # Load data for all microstates and create brain images
            brain_images = []
            valid_microstates = []

            for idx, microstate_label in enumerate(self.comet.micro_labels):
                print(
                    f"Processing microstate {microstate_label} ({idx + 1}/{len(self.comet.micro_labels)})"
                )
                microstate_data = self._load_avg_data(all_available_subjects, idx)

                if microstate_data is not None:
                    # Create brain image for this microstate
                    brain_img = self._create_single_brain_image(microstate_data, microstate_label)
                    if brain_img is not None:
                        brain_images.append((microstate_label, brain_img))
                        valid_microstates.append(microstate_label)
                        print(f"Successfully created brain image for microstate {microstate_label}")
                    else:
                        print(f"Failed to create brain image for microstate {microstate_label}")
                else:
                    print(f"Failed to load data for microstate {microstate_label}")

            if not brain_images:
                print("No valid microstate images created")
                return

            print(f"Creating grid visualization for {len(valid_microstates)} microstates")

            # Create grid layout
            combined_image = self._create_grid_layout(brain_images, len(all_available_subjects))

            # Display the combined image in PyVista plotter
            plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10)
            tex = pv.numpy_to_texture(combined_image)
            self.plotter.add_mesh(plane, texture=tex, name="brain_all_microstates")

            self.plotter.view_xy()
            self.plotter.camera.zoom(1.0)
            self.plotter.update()

            print(
                f"Visualization complete! Showing all microstates averaged across {len(all_available_subjects)} subjects"
            )

        except Exception as e:
            print(f"Error in show_all_microstate_sources: {str(e)}")
            traceback.print_exc()

    def _create_single_brain_image(self, microstate_data, microstate_label):
        """Create a single brain image for a microstate.

        Args:
          microstate_data (np.ndarray): Source-strength array per vertex.
          microstate_label (str): Label used for logging.

        Returns:
          np.ndarray | None: RGB image array or None on failure.
        """
        try:
            # Normalize the data
            data_min = np.min(microstate_data)
            data_max = np.max(microstate_data)

            if data_max > data_min:
                microstate_data_norm = (microstate_data - data_min) / (data_max - data_min)
            else:
                microstate_data_norm = np.zeros_like(microstate_data)

            # Create MNE SourceEstimate object
            n_vertices_total = len(microstate_data_norm)
            n_vertices_per_hemi = n_vertices_total // 2

            vertices = [
                np.arange(n_vertices_per_hemi),  # Left hemisphere
                np.arange(n_vertices_per_hemi),  # Right hemisphere
            ]

            stc_data = microstate_data_norm[:, np.newaxis]

            mystc = mne.SourceEstimate(
                stc_data, vertices=vertices, tmin=0, tstep=1, subject="fsaverage"
            )

            # Plot the brain
            brain = mystc.plot(
                subjects_dir=self.subjects_dir,
                initial_time=0,
                views="lateral",
                cortex="low_contrast",
                hemi="both",
                surface="inflated",
                time_viewer=False,
                background="white",
                size=(400, 300),
                colormap="hot",
                smoothing_steps=10,
                colorbar=False,
                clim=dict(kind="percent", lims=[70, 85, 99]),
            )

            # Take screenshot and close
            img = brain.screenshot()
            brain.close()

            return img

        except Exception as e:
            print(f"Error creating brain image for {microstate_label}: {e}")
            return None

    @staticmethod
    def _create_grid_layout(brain_images, n_subjects):
        """Create a grid layout from brain images.

        Args:
          brain_images (list[tuple[str, np.ndarray]]): (label, image) tuples.
          n_subjects (int): Number of subjects averaged.

        Returns:
          np.ndarray: Combined grid image (RGB).
        """
        try:
            n_microstates = len(brain_images)
            n_cols = min(4, n_microstates)  # Maximum 4 columns
            n_rows = (n_microstates + n_cols - 1) // n_cols

            # Get dimensions from first image
            sample_img = brain_images[0][1]
            img_height, img_width = sample_img.shape[:2]

            # Add padding for titles
            title_height = 50
            margin = 20

            # Calculate grid dimensions
            grid_width = n_cols * img_width + (n_cols + 1) * margin
            grid_height = (
                n_rows * (img_height + title_height) + (n_rows + 1) * margin + 100
            )  # Extra space for main title

            # Create white background
            grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

            # Try to use PIL for better text rendering
            try:
                use_pil = PIL_AVAILABLE
                grid_pil = Image.fromarray(grid_image)
                draw = ImageDraw.Draw(grid_pil)

                # Try to load a font
                try:
                    title_font = ImageFont.truetype("arial.ttf", 20)
                    subtitle_font = ImageFont.truetype("arial.ttf", 16)
                except Exception:
                    try:
                        title_font = ImageFont.truetype("DejaVuSans.ttf", 20)
                        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 16)
                    except Exception:
                        title_font = ImageFont.load_default()
                        subtitle_font = ImageFont.load_default()

            except ImportError:
                use_pil = False
                print("PIL not available, using basic text rendering")

            # Add main title
            main_title = f"All Microstate Sources - Averaged across {n_subjects} subjects"
            if use_pil:
                title_bbox = draw.textbbox((0, 0), main_title, font=title_font)
                title_width = title_bbox[2] - title_bbox[0]
                title_x = (grid_width - title_width) // 2
                draw.text((title_x, 20), main_title, fill=(0, 0, 0), font=title_font)

            # Place brain images in grid
            for idx, (microstate_label, img) in enumerate(brain_images):
                row = idx // n_cols
                col = idx % n_cols

                # Calculate position
                x = margin + col * (img_width + margin)
                y = (
                    80 + margin + row * (img_height + title_height + margin)
                )  # 80 for main title space

                # Place image
                if img.shape[2] == 4:  # RGBA to RGB
                    img = img[:, :, :3]

                grid_image[y : y + img_height, x : x + img_width] = img

                # Add microstate label
                label_text = f"Microstate {microstate_label}"
                if use_pil:
                    label_bbox = draw.textbbox((0, 0), label_text, font=subtitle_font)
                    label_width = label_bbox[2] - label_bbox[0]
                    label_x = x + (img_width - label_width) // 2
                    label_y = y + img_height + 5
                    draw.text((label_x, label_y), label_text, fill=(0, 0, 0), font=subtitle_font)

            if use_pil:
                grid_image = np.array(grid_pil)

            return grid_image

        except Exception as e:
            print(f"Error creating grid layout: {e}")
            # Fallback: just return the first brain image
            if brain_images:
                return brain_images[0][1]
            return np.ones((400, 400, 3), dtype=np.uint8) * 255

    def show_all_microstate_sources_interactive(self):
        """Show all microstate sources in an interactive 3D view with cycling."""
        try:
            selected_items = self.ui.subjects_list.selectedItems()
            if not selected_items:
                return

            selected_subjects = [item.text() for item in selected_items]
            self.plotter.clear()

            # Load data for all microstates
            all_microstate_data = {}

            for idx, microstate_label in enumerate(self.comet.micro_labels):
                microstate_data = self._load_avg_data(selected_subjects, idx)
                if microstate_data is not None:
                    all_microstate_data[microstate_label] = microstate_data

            if not all_microstate_data:
                return

            # Get brain meshes
            meshes = self.source_visualizer.export_meshes()

            if len(meshes) < 2:
                return

            # Create a cycling display of all microstates
            microstate_labels = list(all_microstate_data.keys())
            current_microstate_idx = 0

            def update_microstate_display(microstate_idx):
                # Clear previous data
                self.plotter.clear()

                microstate_label = microstate_labels[microstate_idx]
                microstate_data = all_microstate_data[microstate_label]

                # Normalize the data
                data_min = np.min(microstate_data)
                data_max = np.max(microstate_data)

                if data_max > data_min:
                    microstate_data_norm = (microstate_data - data_min) / (data_max - data_min)
                else:
                    microstate_data_norm = np.zeros_like(microstate_data)

                # Split data for hemispheres
                n_vertices_per_hemi = len(microstate_data_norm) // 2
                lh_data = microstate_data_norm[:n_vertices_per_hemi]
                rh_data = microstate_data_norm[n_vertices_per_hemi:]

                # Add meshes with data
                lh_mesh = meshes[0].copy()
                lh_mesh["data"] = lh_data
                self.plotter.add_mesh(
                    lh_mesh,
                    scalars="data",
                    cmap="hot",
                    clim=[0, 1],
                    opacity=0.9,
                    name="left_hemisphere",
                )

                rh_mesh = meshes[1].copy()
                rh_mesh["data"] = rh_data
                self.plotter.add_mesh(
                    rh_mesh,
                    scalars="data",
                    cmap="hot",
                    clim=[0, 1],
                    opacity=0.9,
                    name="right_hemisphere",
                )

                # Add title and info
                title_text = f"Microstate {microstate_label} ({microstate_idx + 1}/{len(microstate_labels)})\nAveraged across {len(selected_subjects)} subjects"
                self.plotter.add_text(title_text, position="upper_left", font_size=12)

                # Add navigation instructions
                nav_text = "Use LEFT/RIGHT arrow keys to navigate between microstates"
                self.plotter.add_text(nav_text, position="lower_left", font_size=10)

                self.plotter.view_isometric()
                self.plotter.camera.zoom(1.3)

                if microstate_idx == 0:  # Only add scalar bar once
                    self.plotter.add_scalar_bar(title="Source Strength", n_labels=5)

            # Set up keyboard callback for navigation
            def key_press_callback(key):
                nonlocal current_microstate_idx
                if key.lower() == "left" and current_microstate_idx > 0:
                    current_microstate_idx -= 1
                    update_microstate_display(current_microstate_idx)
                elif key.lower() == "right" and current_microstate_idx < len(microstate_labels) - 1:
                    current_microstate_idx += 1
                    update_microstate_display(current_microstate_idx)

            # Add key callback
            self.plotter.add_key_event(key_press_callback)

            # Show first microstate
            update_microstate_display(current_microstate_idx)
            self.plotter.update()

        except Exception:
            pass

    def _load_tess_data(self, selected_subjects, microstate_idx, result_type):
        """Load and average TESS data across subjects.

        Args:
          selected_subjects (list[str]): Subject IDs.
          microstate_idx (int): Index of the microstate.
          result_type (str): 'filtered' or 'raw'.

        Returns:
          np.ndarray | None: Averaged per-vertex array or None.
        """
        all_data = []
        for subject_name in selected_subjects:
            tess_path = os.path.join(self.comet.localized_sources_path, "tess_sources")
            data = self.source_io.load_tess_results(tess_path, subject_name, result_type)
            if data is not None:
                all_data.append(data)

        if all_data:
            avg_data = np.mean(all_data, axis=0)
            if microstate_idx < avg_data.shape[1]:
                return avg_data[:, microstate_idx]

        return None

    def _load_avg_data(self, selected_subjects, microstate_idx):
        """Load and average microstate source data across subjects.

        Args:
          selected_subjects (list[str]): Subject IDs.
          microstate_idx (int): Index of the microstate.

        Returns:
          np.ndarray | None: Averaged per-vertex array or None.
        """
        if microstate_idx < len(self.comet.micro_labels):
            microstate_label = self.comet.micro_labels[microstate_idx]
        else:
            microstate_label = str(microstate_idx)

        all_subject_data = []

        for subject_name in selected_subjects:
            file_path = self._get_microstate_file_path(subject_name, microstate_label)
            data = self._load_microstate_data(file_path)

            if data is not None:
                if data.ndim == 2:
                    data = np.mean(data, axis=0)
                all_subject_data.append(data)

        if all_subject_data:
            # Ensure all data arrays have the same shape
            shapes = [data.shape for data in all_subject_data]
            if len(set(shapes)) > 1:
                min_shape = tuple(min(dim) for dim in zip(*shapes))
                all_subject_data = [data[: min_shape[0]] for data in all_subject_data]

            return np.mean(all_subject_data, axis=0)

        return None

    def _get_microstate_file_path(self, subject_name, microstate_label):
        """Get the file path for a specific subject and microstate.

        Args:
          subject_name (str): Subject identifier.
          microstate_label (str): Microstate label.

        Returns:
          str: Absolute path to the .npy file.
        """
        avg_base_path = os.path.join(self.comet.localized_sources_path, "avg_sources")
        subject_path = os.path.join(avg_base_path, subject_name)
        return os.path.join(subject_path, f"{subject_name}_{microstate_label}.npy")

    @staticmethod
    def _load_microstate_data(file_path):
        """Load microstate data from file.

        Args:
          file_path (str): Path to a .npy array file.

        Returns:
          np.ndarray | None: Loaded array or None if invalid/missing.
        """
        if not os.path.exists(file_path):
            return None

        try:
            data = np.load(file_path)
            # Basic validation
            if np.isnan(data).any() or np.isinf(data).any():
                return None
            return data
        except Exception:
            return None

    def export_all_microstate_views(self, output_dir=None):
        """Export individual brain views for each microstate.

        Args:
          output_dir (str | None): Destination folder; created if missing.
        """
        try:
            if output_dir is None:
                output_dir = os.path.join(self.comet.localized_sources_path, "exported_views")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Use all available subjects for export
            all_available_subjects = self._find_all_subjects_with_avg_data()
            if not all_available_subjects:
                print("No subjects found with averaged source data for export")
                return

            for idx, microstate_label in enumerate(self.comet.micro_labels):
                microstate_data = self._load_avg_data(all_available_subjects, idx)
                if microstate_data is None:
                    continue

                # Normalize the data
                data_min = np.min(microstate_data)
                data_max = np.max(microstate_data)

                if data_max > data_min:
                    microstate_data_norm = (microstate_data - data_min) / (data_max - data_min)
                else:
                    microstate_data_norm = np.zeros_like(microstate_data)

                # Create MNE SourceEstimate object
                n_vertices_total = len(microstate_data_norm)
                n_vertices_per_hemi = n_vertices_total // 2

                vertices = [
                    np.arange(n_vertices_per_hemi),  # Left hemisphere
                    np.arange(n_vertices_per_hemi),  # Right hemisphere
                ]

                stc_data = microstate_data_norm[:, np.newaxis]

                mystc = mne.SourceEstimate(
                    stc_data, vertices=vertices, tmin=0, tstep=1, subject="fsaverage"
                )

                # Plot and save multiple views
                views = ["lateral", "medial", "rostral", "caudal", "dorsal", "ventral"]

                for view in views:
                    brain = mystc.plot(
                        subjects_dir=self.subjects_dir,
                        initial_time=0,
                        views=view,
                        cortex="low_contrast",
                        hemi="both",
                        surface="inflated",
                        time_viewer=False,
                        background="white",
                        size=(800, 600),
                        colormap="hot",
                        smoothing_steps=10,
                        colorbar=True,
                        clim=dict(kind="percent", lims=[70, 85, 99]),
                    )

                    filename = f"microstate_{microstate_label}_{view}_avg_{len(all_available_subjects)}subj.png"
                    filepath = os.path.join(output_dir, filename)
                    brain.save_image(filepath)
                    brain.close()

        except Exception:
            pass

    def show_sources_interactive_3d(self):
        """Show sources using PyVista 3D interactive visualization."""
        try:
            selected_items = self.ui.subjects_list.selectedItems()
            if not selected_items:
                return

            subject_name = selected_items[0].text()
            time_point = float(self.ui.stc_time_input.text()) / 1000.0

            stc_files_dict = self.source_io.find_stc_files(
                self.comet.localized_sources_path, subject_name
            )
            if subject_name not in stc_files_dict or not stc_files_dict[subject_name]:
                return

            stc = self.source_io.read_single_stc(stc_files_dict[subject_name][0])
            if stc is None:
                return

            self.plotter.clear()

            meshes = self.source_visualizer.export_meshes()
            time_idx = np.abs(stc.times - time_point).argmin()
            data_timepoint = stc.data[:, time_idx]

            n_vertices_per_hemi = len(data_timepoint) // 2
            lh_data = data_timepoint[:n_vertices_per_hemi]
            rh_data = data_timepoint[n_vertices_per_hemi:]

            all_data = np.concatenate([lh_data, rh_data])
            data_min, data_max = np.min(all_data), np.max(all_data)
            if data_max > data_min:
                lh_data = (lh_data - data_min) / (data_max - data_min)
                rh_data = (rh_data - data_min) / (data_max - data_min)

            if len(meshes) >= 2:
                lh_mesh = meshes[0]
                lh_mesh["data"] = lh_data
                self.plotter.add_mesh(
                    lh_mesh,
                    scalars="data",
                    cmap="hot",
                    clim=[0, 1],
                    opacity=0.9,
                    name="left_hemisphere",
                )

                rh_mesh = meshes[1]
                rh_mesh["data"] = rh_data
                self.plotter.add_mesh(
                    rh_mesh,
                    scalars="data",
                    cmap="hot",
                    clim=[0, 1],
                    opacity=0.9,
                    name="right_hemisphere",
                )

            self.plotter.add_text(
                f"Subject: {subject_name} | Time: {time_point:.3f}s",
                position="upper_left",
                font_size=12,
            )

            self.plotter.view_isometric()
            self.plotter.camera.zoom(1.3)
            self.plotter.add_scalar_bar(title="Source Strength", n_labels=5)
            self.plotter.update()

        except Exception:
            pass

    def export_current_view(self, filename=None):
        """Export the current view as an image.

        Args:
          filename (str | None): Output image file name. If None, a timestamped
            name is generated in the working directory.
        """
        try:
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"brain_view_{timestamp}.png"

            self.plotter.screenshot(filename)

        except Exception:
            pass

    def reset_view(self):
        """Reset the camera view to default."""
        try:
            self.plotter.reset_camera()
            self.plotter.view_isometric()
        except Exception:
            pass

    def clear_plotter(self):
        """Clear all objects from the plotter."""
        with contextlib.suppress(Exception):
            self.plotter.clear()

    def closeEvent(self, event):
        """Handle window close event.

        Args:
          event (QCloseEvent): Qt close event.
        """
        try:
            if hasattr(self, "plotter"):
                self.plotter.close()
            event.accept()
        except Exception:
            event.accept()
