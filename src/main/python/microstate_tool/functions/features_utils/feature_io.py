"""
This class provides methods for exporting and importing calculated features_utils to and from files.
It supports exporting features_utils in both 'static' and 'dynamic' modes, and importing features_utils from various file formats.
"""

import os.path
import pandas as pd
import numpy as np
import json

class FeatureIO:
    @staticmethod
    def export_features(features, mode, output_folder, export_format='csv', filename='features_utils'):
        """
        Export the calculated features_utils to the specified file format.

        Args:
            features (dict or list of dict): Results obtained from feature extraction methods.
            mode (str): The mode used for feature extraction ('static' or 'dynamic').
            output_folder (str):
            export_format (str): Format for exporting ('csv', 'pkl', 'hdf', or 'json').
            filename (str): Name of the output file without extension.
        """
        if mode == 'static':
            if export_format == 'csv':
                df = pd.DataFrame(features.items(), columns=['Element', 'Value'])
                df.to_csv(os.path.join(output_folder, f'{filename}_static.csv'), index=False)
            elif export_format == 'pkl':
                pd.Series(features).to_pickle(os.path.join(output_folder, f'{filename}_static.pkl'))
            elif export_format == 'hdf':
                pd.Series(features).to_hdf(os.path.join(output_folder, f'{filename}_static.h5'), key='features_utils')
            elif export_format == 'json':
                with open(os.path.join(output_folder, f'{filename}_static.json'), 'w') as f:
                    json.dump(features, f)
            else:
                raise ValueError("Invalid export format.")

        elif mode == 'dynamic':
            if export_format == 'csv':
                dfs = []
                for idx, window_features in enumerate(features):
                    df = pd.DataFrame(window_features.items(), columns=['Element', f'Value_Window_{idx + 1}'])
                    dfs.append(df)
                combined_df = pd.concat(dfs, axis=0, ignore_index=True)
                combined_df.to_csv(os.path.join(output_folder, f'{filename}_dynamic.csv'), index=False)
            elif export_format == 'pkl':
                with open(os.path.join(output_folder, f'{filename}_dynamic.pkl'), 'wb') as f:
                    np.save(f, features)
            elif export_format == 'hdf':
                combined_series = pd.concat([pd.Series(window_features) for window_features in features], axis=1)
                combined_series.to_hdf(os.path.join(output_folder, f'{filename}_dynamic.h5'), key='features_utils')
            elif export_format == 'json':
                with open(os.path.join(output_folder, f'{filename}_dynamic.json'), 'w') as f:
                    json.dump(features, f)
            else:
                raise ValueError("Invalid export format.")
        else:
            raise ValueError("Invalid mode. Supported modes are 'static' and 'dynamic'.")

    @staticmethod
    def import_features(file_path, import_format='csv'):
        """
        Import the calculated features_utils from a file.

        Args:
            file_path (str): Path to the input file.
            import_format (str): Format of the imported file ('csv', 'pkl', 'hdf', or 'json').

        Returns:
            dict or list of dict: Imported features_utils.
        """
        if import_format == 'csv':
            df = pd.read_csv(file_path)
            features = dict(zip(df['Element'], df['Value']))
        elif import_format == 'pkl':
            features = pd.read_pickle(file_path)
        elif import_format == 'hdf':
            hdf = pd.read_hdf(file_path)
            features = hdf.to_dict()
        elif import_format == 'json':
            with open(file_path, 'r') as f:
                features = json.load(f)
        else:
            raise ValueError("Invalid import format.")

        return features
