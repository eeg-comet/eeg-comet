"""
Last Modified: April 18th, 2023
Description: This file defines a function for plotting extracted features using plotly.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

import pandas as pd
import plotly.express as px


def plot_features(features_path):
    """
    Plots extracted features using plotly.

    Inputs:
        features_path (string): The path to the extracted features file.

    Outputs:
        None
    """
    # Load the extracted features file
    features = pd.read_csv(features_path)

    # Reshape the dataframe suitable for statsmodels package
    df_features = pd.melt(features.reset_index(), id_vars=['Filename'], value_vars=['FOC_A', 'FOC_B', 'FOC_C', 'FOC_D', 'FOC_E'])

    # Replace column names
    df_features.columns = ['Filename', 'Feature', 'FOC']
    df_features['FOC'] = df_features['FOC']

    # Create a box plot using plotly
    fig = px.box(df_features, x="Filename", y="FOC")
    fig.show()
