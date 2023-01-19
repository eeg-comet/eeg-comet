
from configparser import ConfigParser


def initialize_config(config_path, config):
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
    config = ConfigParser()
    config.read(config_path)
    return config


def save_config(config_path, config):
    with open(config_path, 'w+') as configfile:
        config.write(configfile)
    return config
