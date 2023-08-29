"""
This script defines a class that facilitates the visualization of various features_utils extracted from sequences of symbols.
It utilizes the FeatureIO class to load exported features_utils and provides methods to visualize these features_utils using
group bar charts and heatmaps.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functions.features.feature_io import FeatureIO

class FeatureVisualizer:
    def __init__(self):
        pass

    def visualize_features(self, feature_path, feature_name, window_index=0, mode='static'):
        """
        Visualize specified features_utils using group bar charts.

        Args:
            feature_path (str): Path to the directory containing exported feature files.
            feature_name (str): Name of the feature to visualize.
            window_index (int or None): Index of the window to visualize. If None, visualizes all windows.
            mode (str): The mode for calculation. 'static' visualizes average features_utils,
                        'dynamic' visualizes features_utils for each window.
        """
        # Create a FeatureIO instance with the provided feature_path
        feature_io = FeatureIO(feature_path)

        feature_results = {}

        feature_results[feature_name] = feature_io.load_features(feature_name, mode)
        print(feature_results)
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))

        window_data = feature_results[feature_name][window_index]

        plt.title(f'{feature_name.replace("_", " ").title()} - Window {window_index + 1}')
        symbols = sorted(set(symbol for symbol in window_data.keys()))
        sns.barplot(x=list(window_data.keys()), y=list(window_data.values()))
        plt.ylabel('Value')
        plt.xlabel('Microstate')

        plt.tight_layout()
        plt.show()

    def visualize_transition_probability(self, feature_path=None, mode='static'):
        """
        Visualize transition probability as a heatmap.

        Args:
            mode (str): The mode for calculation. 'static' visualizes average transition probability,
                        'dynamic' visualizes transition probability for each window.
            feature_path (str or None): Path to the directory containing exported feature files.
        """
        # Create a FeatureIO instance with the provided feature_path
        feature_io = FeatureIO(feature_path)

        transition_prob_result = feature_io.load_features('transition_probability', mode)

        if mode == 'static':
            title = 'Average Transition Probability'
        else:
            title = 'Transition Probability (Dynamic)'

        if mode == 'dynamic':
            for idx, window_results in enumerate(transition_prob_result):
                if not window_results:  # Handle empty window results
                    print(f"No transition probability data for Window {idx + 1}.")
                    continue

                symbols = sorted(set(symbol for pair in window_results for symbol in pair))
                num_symbols = len(symbols)
                transition_matrix = np.zeros((num_symbols, num_symbols))

                for from_symbol, to_symbol in window_results:
                    from_idx = symbols.index(from_symbol)
                    to_idx = symbols.index(to_symbol)
                    transition_matrix[from_idx, to_idx] = window_results[(from_symbol, to_symbol)]

                fig, ax = plt.subplots()
                cax = ax.matshow(transition_matrix, cmap="YlGnBu")
                plt.title(f'{title} - Window {idx + 1}')
                plt.xlabel('To Symbol')
                plt.ylabel('From Symbol')
                plt.xticks(range(num_symbols), symbols)
                plt.yticks(range(num_symbols), symbols)
                plt.colorbar(cax)
                plt.show()
        else:
            symbols = sorted(set(symbol for pair in transition_prob_result for symbol in pair))
            num_symbols = len(symbols)
            transition_matrix = np.zeros((num_symbols, num_symbols))

            for from_symbol, to_symbol in transition_prob_result:
                from_idx = symbols.index(from_symbol)
                to_idx = symbols.index(to_symbol)
                transition_matrix[from_idx, to_idx] = transition_prob_result[(from_symbol, to_symbol)]

            fig, ax = plt.subplots()
            cax = ax.matshow(transition_matrix, cmap="YlGnBu")
            plt.title(title)
            plt.xlabel('To Symbol')
            plt.ylabel('From Symbol')
            plt.xticks(range(num_symbols), symbols)
            plt.yticks(range(num_symbols), symbols)
            plt.colorbar(cax)
            plt.show()

