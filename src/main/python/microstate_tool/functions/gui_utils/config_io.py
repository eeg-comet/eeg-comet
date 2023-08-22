
from configparser import ConfigParser

def initialize_config(config_path, config):
    """
    Initialize and populate a configuration dictionary with default values.

    Parameters:
    config_path (str): The file path where the configuration will be saved.
    config (dict): An empty dictionary where configuration settings will be stored.

    Returns:
    None

    Explanation:
    This function initializes a configuration dictionary with predefined sections and default values.
    It sets up various sections for different aspects of a data processing pipeline and assigns initial values.
    The purpose is to provide a structured way to manage and store settings for data processing steps.

    Example usage:
    >>> config = {}  # An empty dictionary to hold configuration settings
    >>> initialize_config('/path/to/save/config.ini', config)
    """

    # Create sections for different processing aspects
    config['progress'] = {}  # Progress tracking
    config['study info'] = {}  # Study information
    config['preprocessing settings'] = {}  # Preprocessing settings
    config['preprocessing results'] = {}  # Preprocessing results
    config['clustering_utils settings'] = {}  # Clustering settings
    config['clustering_utils results'] = {}  # Clustering results
    config['backfitting_utils settings'] = {}  # Backfitting settings
    config['feature extraction settings'] = {}  # Feature extraction settings
    config['feature visualization groups'] = {}  # Feature visualization groups
    config['source localization settings'] = {}  # Source localization settings

    # Initialize progress indicators with 'False'
    str_false = "False"
    config['progress']['done_preprocessing'] = str_false
    config['progress']['done_clustering'] = str_false
    config['progress']['done_labeling_microstates'] = str_false
    config['progress']['done_backfitting'] = str_false
    config['progress']['done_extracting_features'] = str_false
    config['progress']['done_extracting_microsegments'] = str_false
    config['progress']['done_source_localization'] = str_false

    # Save the initialized configuration to the specified file path
    save_config(config_path, config)


def load_config(config_path):
    """
    Load and parse configuration settings from a file.

    Parameters:
    config_path (str): The path to the configuration file.

    Returns:
    config (ConfigParser): A ConfigParser object containing the loaded configuration settings.

    Explanation:
    This function reads and parses configuration settings from a specified file using the ConfigParser module.
    The ConfigParser object can be used to access and manipulate the loaded configuration values.

    Example usage:
    >>> loaded_config = load_config('/path/to/config.ini')
    >>> value = loaded_config.get('section', 'option')
    """

    config = ConfigParser()  # Create a ConfigParser object
    config.read(config_path)  # Read and parse the configuration file
    return config


def save_config(config_path, config):
    """
    Save configuration settings to a file.

    Parameters:
    config_path (str): The path where the configuration file will be saved.
    config (ConfigParser): A ConfigParser object containing the configuration settings.

    Returns:
    config (ConfigParser): The same ConfigParser object provided as input.

    Explanation:
    This function saves configuration settings from a ConfigParser object to a specified file.
    The ConfigParser object should contain the desired configuration values.

    Example usage:
    >>> config = ConfigParser()
    >>> config['section']['option'] = 'value'
    >>> save_config('/path/to/save/config.ini', config)
    """

    with open(config_path, 'w+') as configfile:
        config.write(configfile)  # Write the configuration settings to the specified file
    return config

