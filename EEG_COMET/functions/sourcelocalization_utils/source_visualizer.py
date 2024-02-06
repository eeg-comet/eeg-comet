
import mne
import os.path
import numpy as np
import pyvista as pv
from functions.data_utils.data_io import DataIO


class SourceVisualizer:
    def __init__(self, subjects_dir, spacing, localized_sources_path):

        self.subjects_dir = subjects_dir
        self.spacing = spacing
        self.localized_sources_path = localized_sources_path
        self.tess_sources_path = os.path.join(localized_sources_path, 'tess_sources')
        self.avg_sources_path = os.path.join(localized_sources_path, 'avg_sources')

    def export_meshes(self):

        src_filename = 'fsaverage-' + self.spacing[:-1] + '-' + self.spacing[-1] + '-src.fif'
        src_filepath = os.path.join(self.subjects_dir, 'fsaverage' ,'bem', src_filename)
        src = mne.read_source_spaces(src_filepath, verbose=False)
        meshes = []
        vertices = np.array(src[0]['rr'])
        triangles = np.array(src[0]['tris'])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh1 = pv.PolyData(vertices, triangles)
        meshes.append(mesh1)
        vertices = np.array(src[1]['rr'])
        triangles = np.array(src[1]['tris'])
        triangles = np.c_[np.full(len(triangles), 3), triangles]
        mesh2 = pv.PolyData(vertices, triangles)
        meshes.append(mesh2)

        return meshes

    @staticmethod
    def normalize_matrix(matrix):
        return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

    def normalize_stc_data(self, stc_data):
        return np.array([self.normalize_matrix(row) for row in stc_data])

    def preprocess_data(self, selected_items, source_mode):
        if source_mode in ["tess_filtered", "tess_raw"]:

            pattern = '*-zscore*' if source_mode == "tess_raw" else '*filtered_zscore*'
            stc_list_path, _ = DataIO().find_data(self.tess_sources_path, extension='.npy', pattern=pattern)
        elif source_mode in ["avg_filtered", "avg_raw"]:
            # TODO: complete this
            stc_list_path, _ = DataIO().find_data(self.avg_sources_path, extension='.npy', pattern='avg')

        selected_subjects = [item.text() for item in selected_items]
        stc_list_selected_paths = [full_path for full_path in stc_list_path if
                                   any(folder in full_path for folder in selected_subjects)]
        print(stc_list_selected_paths)

        stc_sum = np.zeros_like(np.load(stc_list_selected_paths[0])) if stc_list_selected_paths else []
        for stc_path in stc_list_selected_paths:
            stc_data = np.load(stc_path)
            stc_data = self.normalize_stc_data(stc_data)
            stc_sum += stc_data

        stc_avg = stc_sum / len(stc_list_selected_paths) if stc_list_selected_paths else np.zeros_like(stc_sum)

        mystc = mne.SourceEstimate(stc_avg,
                                   vertices=[np.arange(stc_avg.shape[0] // 2), np.arange(stc_avg.shape[0] // 2)],
                                   tmin=0, tstep=1, subject='fsaverage')

        return mystc, stc_avg

    def plot_sources(self, selected_items, source_mode, initial_time, spacing):
        # [TODO] Need to add plots within the main visualization window

        [mystc, stc_avg] = self.preprocess_data(selected_items, source_mode)

        print(mystc)
        print(stc_avg.shape)
        # Initial time refers to which microstate is displayed (i.e. 0 = microstate A, 6 = microstate G)

        print(initial_time)
        # """
        brain = mystc.plot(subjects_dir=self.subjects_dir,
                           initial_time=initial_time,
                           views=['lateral', 'medial'],
                           cortex='low_contrast',
                           hemi='split',
                           surface='inflated',
                           clim=dict(kind='value', lims=[0, 0.5, 1]),
                           time_viewer=False,
                           background='white',
                           spacing=spacing,
                           size=(800, 800),
                           colormap='jet',
                           smoothing_steps=15,
                           colorbar=True
                           )
        # """

        # brain.add_annotation("HCPMMP1_combined", borders=2)
        # brain.save_image(filename=os.path.join(parent_path, f'avg_filtered_zscore_M_{initial_time}.png'))
