"""
Last Modified: April 18th, 2023

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

from configparser import ConfigParser


def initialize_config(config_path, config):
    """
    Description: Initializes a configuration file with default sections and settings.

    Inputs:
        config_path (string): The path to the configuration file to be initialized.
        config (ConfigParser object): A ConfigParser object representing the configuration file.

    Outputs:
        None
    """
    # Create sections
    config['progress'] = {}
    config['study info'] = {}
    config['preprocessing settings'] = {}
    config['preprocessing results'] = {}
    config['clustering settings'] = {}
    config['clustering results'] = {}
    config['backfitting settings'] = {}
    config['feature extraction settings'] = {}
    config['feature visualization groups'] = {}
    config['source localization settings'] = {}
    # Initialize progress
    config['progress']['done_preprocessing'] = str(False)
    config['progress']['done_clustering'] = str(False)
    config['progress']['done_labeling_microstates'] = str(False)
    config['progress']['done_backfitting'] = str(False)
    config['progress']['done_extracting_features'] = str(False)
    config['progress']['done_extracting_microsegments'] = str(False)
    config['progress']['done_source_localization'] = str(False)
    # Write to config
    save_config(config_path, config)


def load_config(config_path):
    """
    Loads a configuration file using the ConfigParser module.

    Inputs:
        config_path (string): Path to the configuration file.

    Outputs:
        config (ConfigParser object): Configuration file
            The configuration file loaded using the ConfigParser module.

    """
    # Create a ConfigParser object
    config = ConfigParser()
    # Read the configuration file
    config.read(config_path)
    # Return the configuration file object
    return config


def save_config(config_path, config):
    """
    Saves a configuration file.

    Inputs:
        config_path (string): Path to the configuration file.
        config (ConfigParser object): The configuration object to save.

    Outputs:
        config (ConfigParser object): The saved configuration object.

    """
    # Open the configuration file for writing
    with open(config_path, 'w+') as configfile:
        # Write the configuration object to the file
        config.write(configfile)
    # Return the saved configuration object
    return config

