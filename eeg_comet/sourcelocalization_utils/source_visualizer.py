"""Source visualization utilities leveraging MNE and PyVista."""

import os.path

import mne
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

from eeg_comet.data_utils.data_io import DataIO


class SourceVisualizer:
    """Visualize source data for subjects and microstates."""

    def __init__(self, subjects_dir, spacing, localized_sources_path):
        """Initialize the SourceVisualizer.

        Args:
            subjects_dir: The directory where the subject-specific MRI data is stored.
            spacing: The spacing parameter for creating the source space.
            localized_sources_path: The directory where the localized source data is stored.
        """
        self.subjects_dir = subjects_dir
        self.spacing = spacing
        self.localized_sources_path = localized_sources_path
        self.tess_sources_path = os.path.join(localized_sources_path, "tess_sources")
        self.avg_sources_path = os.path.join(localized_sources_path, "avg_sources")

    def export_meshes(self):
        """Export the source space meshes."""
        src_filename = f"fsaverage-{self.spacing[:-1]}-{self.spacing[-1]}-src.fif"
        src_filepath = os.path.join(self.subjects_dir, "fsaverage", "bem", src_filename)
        src = mne.read_source_spaces(src_filepath, verbose=False)
        vertices = np.array(src[0]["rr"])
        triangles = np.array(src[0]["tris"])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh1 = pv.PolyData(vertices, triangles)
        meshes = [mesh1]
        vertices = np.array(src[1]["rr"])
        triangles = np.array(src[1]["tris"])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh2 = pv.PolyData(vertices, triangles)
        meshes.append(mesh2)

        return meshes

    @staticmethod
    def normalize_matrix(matrix):
        """Normalize a vector to [0, 1] range."""
        return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

    def normalize_stc_data(self, stc_data):
        """Normalize each row of STC data to [0, 1]."""
        return np.array([self.normalize_matrix(row) for row in stc_data])

    def preprocess_data(self, selected_items, source_mode):
        """Prepare averaged STC data for selected subjects based on mode."""
        if source_mode in ["tess_filtered", "tess_raw"]:

            pattern = "*-zscore*" if source_mode == "tess_raw" else "*filtered_zscore*"
            stc_list_path, _ = DataIO().find_data(
                self.tess_sources_path, extension=".npy", pattern=pattern
            )
        elif source_mode in ["avg_filtered", "avg_raw"]:
            stc_list_path, _ = DataIO().find_data(
                self.avg_sources_path, extension=".npy", pattern="avg"
            )

        selected_subjects = [item.text() for item in selected_items]
        stc_list_selected_paths = [
            full_path
            for full_path in stc_list_path
            if any(folder in full_path for folder in selected_subjects)
        ]
        print(stc_list_selected_paths)

        stc_sum = (
            np.zeros_like(np.load(stc_list_selected_paths[0])) if stc_list_selected_paths else []
        )
        for stc_path in stc_list_selected_paths:
            stc_data = np.load(stc_path)
            stc_data = self.normalize_stc_data(stc_data)
            stc_sum += stc_data

        stc_avg = (
            stc_sum / len(stc_list_selected_paths)
            if stc_list_selected_paths
            else np.zeros_like(stc_sum)
        )

        mystc = mne.SourceEstimate(
            stc_avg,
            vertices=[np.arange(stc_avg.shape[0] // 2), np.arange(stc_avg.shape[0] // 2)],
            tmin=0,
            tstep=1,
            subject="fsaverage",
        )

        return mystc, stc_avg

    def plot_stc(self, subject, stc, time_point):
        """Plot STC data at a specific time point on a brain surface."""
        # Find the closest time point index
        time_idx = np.abs(stc.times - time_point).argmin()
        selected_time = stc.times[time_idx]

        print(f"Displaying time point: {selected_time:.3f}s (requested: {time_point:.3f}s)")

        # Handle potential data scaling issues that could cause overflow
        data_max = np.max(np.abs(stc.data))
        if data_max > 1e6:
            print(
                f"Warning: Large data values detected (max: {data_max:.2e}). Consider scaling your data."
            )

        # Safe color limits based on data percentiles
        vmin, vmid, vmax = np.percentile(stc.data, [70, 85, 99])

        stc.plot(
            subject=subject,
            subjects_dir=self.subjects_dir,
            hemi="both",
            time_viewer=False,  # Disable time viewer to avoid slider issues
            views="lateral",
            initial_time=selected_time,  # Use the selected time point
            background="white",
            size=(800, 600),
            smoothing_steps=10,
            time_unit="ms",
            clim=dict(kind="value", lims=[vmin, vmid, vmax]),  # Set explicit color limits
        )

    def plot_sources(self, selected_items, source_mode, initial_time, spacing):
        """Plot the source data.

        Args:
            selected_items: The selected items.
            source_mode: The source mode.
            initial_time: The initial time.
            spacing: The spacing parameter.

        Returns:
            None
        """
        [mystc, stc_avg] = self.preprocess_data(selected_items, source_mode)

        print(mystc)
        print(stc_avg.shape)
        # Initial time refers to which microstate is displayed (i.e. 0 = microstate A, 6 = microstate G)

        print(initial_time)
        # """
        # 3D
        brain = mystc.plot(
            subjects_dir=self.subjects_dir,
            initial_time=initial_time,
            views=["lateral", "medial"],
            cortex="low_contrast",
            hemi="split",
            surface="inflated",
            clim=dict(kind="value", lims=[0, 0.5, 1]),
            time_viewer=False,
            background="white",
            spacing=spacing,
            size=(800, 800),
            colormap="jet",
            smoothing_steps=15,
            colorbar=True,
        )
        # """
        # 2D
        brain.add_text(0.1, 0.9, "", "title", font_size=16)
        img = brain.screenshot()
        brain.close()
        plt.figure()
        plt.imshow(img)
        plt.axis("off")

        # brain.add_annotation("HCPMMP1_combined", borders=2)
        # brain.save_image(filename=os.path.join(parent_path, f'avg_filtered_zscore_M_{initial_time}.png'))
