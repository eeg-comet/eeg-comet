
import os.path
import pandas as pd


class FeatureIO:
    """
    The FeatureIO class provides methods for exporting and importing calculated features.
    """

    @staticmethod
    def export_features(features_df, feature_type, feature_mode, output_folder, export_format='.csv'):
        """
        Export calculated features to the specified file format.

        Args:
            features_df (pandas.DataFrame): The DataFrame containing the calculated features.
            feature_type (str): The type of the features.
            feature_mode (str): The mode of the features.
            output_folder (str): The folder path to save the exported file.
            export_format (str, optional): The format of the exported file. Defaults to '.csv'.
        """

        output_path = os.path.join(output_folder, f'{feature_type}_{feature_mode}_features{export_format}')

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
            file_path (str): The path to the file containing the calculated features.
            import_format (str, optional): The format of the imported file. Defaults to '.csv'.

        Returns:
            pandas.DataFrame: The imported DataFrame containing the calculated features.
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
