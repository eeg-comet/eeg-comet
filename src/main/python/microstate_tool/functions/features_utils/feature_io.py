"""
This class provides methods for exporting and importing calculated features_utils to and from files.
It supports exporting features_utils in both 'static' and 'dynamic' modes, and importing features_utils from various file formats.
"""

import os.path
import pandas as pd

class FeatureIO:
    @staticmethod
    def export_features(features_df, mode, output_folder, export_format='.csv'):
        """
        Export calculated features to the specified file format.

        Args:
            features_df (DataFrame): DataFrame containing extracted features.
            mode (str): Feature extraction mode ('static' or 'dynamic').
            output_folder (str): Path to the output folder.
            export_format (str): Format for exporting ('.csv', '.pkl', '.hdf', or '.json').
        """
        output_path = os.path.join(output_folder, f'{mode}_features{export_format}')

        if export_format == '.csv':
            features_df.to_csv(output_path, index=False)
        elif export_format == '.pkl':
            features_df.to_pickle(output_path)
        elif export_format == '.hdf':
            features_df.to_hdf(output_path, key='features', mode='w')
        elif export_format == '.json':
            features_df.to_json(output_path, orient='records', lines=True)
        else:
            raise ValueError("Invalid export format.")

    @staticmethod
    def import_features(file_path, import_format='.csv'):
        """
        Import calculated features from a file.

        Args:
            file_path (str): Path to the input file.
            import_format (str): Format of the imported file ('.csv', '.pkl', '.hdf', or '.json').

        Returns:
            DataFrame: Imported features.
        """
        if import_format == '.csv':
            features = pd.read_csv(file_path)
        elif import_format == '.pkl':
            features = pd.read_pickle(file_path)
        elif import_format == '.hdf':
            features = pd.read_hdf(file_path, key='features')
        elif import_format == '.json':
            features = pd.read_json(file_path, orient='records', lines=True)
        else:
            raise ValueError("Invalid import format.")

        return features
