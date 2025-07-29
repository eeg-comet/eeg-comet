import os
from datetime import datetime
from typing import Optional, Callable


class EEGCometLogger:
    """
    Comprehensive logging utility for EEG-COMET with consistent emoji patterns.
    """
    
    def __init__(self, log_window=None):
        """
        Initialize the logger.
        
        Parameters
        ----------
        log_window : Optional object
            LogWindow instance for GUI logging
        """
        self.log_window = log_window
        self._step_prefixes = {
            'PREPROCESSING': '⌛  [PREPROCESSING]',
            'CLUSTERING': '⌛  [CLUSTERING]',
            'LABELING': '☄️  [LABELING]',
            'BACKFITTING': '⌛  [BACKFITTING]',
            'FEATURE_EXTRACTION': '⌛  [FEATURE EXTRACTION]',
            'SOURCE_LOCALIZATION': '⌛  [SOURCE LOCALIZATION]',
            'OPTIMIZATION': '⌛  [OPTIMIZATION]',
            'COREGISTRATION': '☄️  [COREGISTRATION]',
            'VISUALIZATION': '☄️  [VISUALIZATION]',
            'DATA_IO': '☄️  [DATA I/O]',
            'CONFIGURATION': 'ℹ️  [CONFIGURATION]',
            'VALIDATION': 'ℹ️  [VALIDATION]',
            'STUDY_LOADING': 'ℹ️  [STUDY LOADING]',
            'SHUTDOWN': '👋  [SHUTDOWN]',
            'ORGANIZATION': '🧠  [ORGANIZATION]',
            'GITHUB': '🌐  [GITHUB]',
            'CONTACT': '✉️  [CONTACT]'
        }
    
    def _get_step_prefix(self, step: str) -> str:
        """Get the emoji prefix for a given step."""
        return self._step_prefixes.get(step.upper(), '☄️  [PROCESSING]')

    @staticmethod
    def _get_info_prefix(step: str) -> str:
        """Get the info emoji prefix for a given step."""
        return f"ℹ️  [{step.replace('_', ' ').upper()}]"

    @staticmethod
    def _print_to_console(message: str):
        """Print message to console."""
        print(message)
    
    def _log_to_gui(self, message: str, log_type: str = 'info'):
        """Log message to GUI if available."""
        if self.log_window is not None and hasattr(self.log_window, 'append_log'):
            self.log_window.append_log(message, log_type=log_type)
    
    def section_header(self, step: str, title: str = None):
        """
        Print a section header with centered title.
        
        Parameters
        ----------
        step : str
            The processing step (e.g., 'PREPROCESSING', 'CLUSTERING')
        title : str, optional
            Custom title to display. If None, uses the step name.
        """
        if title is None:
            title = step.replace('_', ' ').title()
        
        # Create centered separator with title
        separator_length = 60
        title_length = len(title)
        padding = (separator_length - title_length) // 2
        
        separator = "=" * separator_length
        title_line = "=" * padding + title + "=" * (separator_length - title_length - padding)
        
        # Print to console
        self._print_to_console(f"\n{separator}")
        self._print_to_console(title_line)
        self._print_to_console(f"{separator}")
        
        # Log to GUI
        self._log_to_gui(f"{title}", log_type='section')
    
    def toolbox_header(self, main_title: str, subtitle: str):
        """
        Print a toolbox header with main title and subtitle in 2 lines with same separator pattern.
        
        Parameters
        ----------
        main_title : str
            The main title (e.g., 'EEG-COMET')
        subtitle : str
            The subtitle (e.g., '(EEG Comprehensive Microstate Extraction Toolbox)')
        """
        separator_length = 60
        
        # Create centered main title with ☄️ symbols
        main_title_with_emoji = f"☄️ {main_title} ☄️"
        main_title_length = len(main_title_with_emoji)
        main_padding = (separator_length - main_title_length) // 2
        main_title_line = "=" * main_padding + main_title_with_emoji + "=" * (separator_length - main_title_length - main_padding)
        
        # Create centered subtitle
        subtitle_length = len(subtitle)
        subtitle_padding = (separator_length - subtitle_length) // 2
        subtitle_line = "=" * subtitle_padding + subtitle + "=" * (separator_length - subtitle_length - subtitle_padding)
        
        separator = "=" * separator_length
        
        # Print to console
        self._print_to_console(f"\n{separator}")
        self._print_to_console(main_title_line)
        self._print_to_console(subtitle_line)
        self._print_to_console(f"{separator}")
        
        # Log to GUI
        self._log_to_gui(f"{main_title} - {subtitle}", log_type='section')
    
    def processing_start(self, step: str, message: str):
        """
        Log the start of a processing step.
        
        Parameters
        ----------
        step : str
            The processing step
        message : str
            The message to display
        """
        # Use ☄️ for "Starting" messages, step-specific prefix for others
        if message.lower().startswith("starting"):
            full_message = f"☄️  [{step.replace('_', ' ').upper()}] {message}..."
        else:
            prefix = self._get_step_prefix(step)
            full_message = f"{prefix} {message}..."
        
        self._print_to_console(full_message)
        self._log_to_gui(message, log_type='process')
    
    def processing_info(self, step: str, message: str):
        """
        Log processing information.
        
        Parameters
        ----------
        step : str
            The processing step
        message : str
            The message to display
        """
        # Use step-specific prefix for SHUTDOWN, ORGANIZATION, GITHUB, CONTACT, and CLUSTERING processing messages
        if step.upper() in ['SHUTDOWN', 'ORGANIZATION', 'GITHUB', 'CONTACT'] or \
           (step.upper() == 'CLUSTERING' and 'Identifying' in message):
            prefix = self._get_step_prefix(step)
        else:
            prefix = self._get_info_prefix(step)
        
        full_message = f"{prefix} {message}"
        
        self._print_to_console(full_message)
        self._log_to_gui(message, log_type='info')
    
    def processing_success(self, step: str, message: str):
        """
        Log successful completion of a processing step.
        
        Parameters
        ----------
        step : str
            The processing step
        message : str
            The message to display
        """
        success_message = f"✅  [{step.replace('_', ' ').upper()}] {message}"
        
        self._print_to_console(success_message)
        self._log_to_gui(message, log_type='success')
    
    def warning(self, step: str, message: str):
        """
        Log a warning message.
        
        Parameters
        ----------
        step : str
            The processing step
        message : str
            The warning message
        """
        warning_message = f"⚠️  [{step.replace('_', ' ').upper()}] {message}"
        
        self._print_to_console(warning_message)
        self._log_to_gui(message, log_type='warning')
    
    def error(self, step: str, message: str):
        """
        Log an error message (app should not crash).
        
        Parameters
        ----------
        step : str
            The processing step
        message : str
            The error message
        """
        error_message = f"❌  [{step.replace('_', ' ').upper()}] {message}"
        
        self._print_to_console(error_message)
        self._log_to_gui(message, log_type='error')
    
    def stop_requested(self, step: str):
        """
        Log when a stop is requested for a processing step.
        
        Parameters
        ----------
        step : str
            The processing step
        """
        stop_message = f"❌  [{step.replace('_', ' ').upper()}] Stop Requested - Please Wait ..."
        
        self._print_to_console(stop_message)
        self._log_to_gui("Stop Requested - Please Wait ...", log_type='warning')
    
    def settings_info(self, step: str, settings_dict: dict):
        """
        Log configuration settings.
        
        Parameters
        ----------
        step : str
            The processing step
        settings_dict : dict
            Dictionary of settings to log
        """
        # Print each setting as individual info lines, but skip redundant clustering info
        for key, value in settings_dict.items():
            # Skip redundant clustering settings that are already shown in the main message
            if step.upper() == "CLUSTERING" and key in ["Number of Maps", "Number of Repeats"]:
                continue
                
            # Special handling for clustering method to show shorter citation in logs
            if key == "Clustering Method" and step.upper() == "CLUSTERING":
                # Map full method names to shorter citation format
                method_mapping = {
                    "Modified K-Means Clustering (Pascual-Marqui et al. 1995)": "Modified K-Means (https://doi.org/10.1109/10.391164/)",
                    "Modified K-Means Clustering with Spatial Similarity": "Modified K-Means with Spatial Similarity",
                    "Topographic Atomize and Agglomerate Hierarchical Clustering": "TAAHC Clustering"
                }
                log_value = method_mapping.get(value, value)
                self.processing_info(step, f"{key}: {log_value}")
            else:
                self.processing_info(step, f"{key}: {value}")
    
    def progress_update(self, step: str, current: int, total: int, message: str = None):
        """
        Log progress updates.
        
        Parameters
        ----------
        step : str
            The processing step
        current : int
            Current progress value
        total : int
            Total progress value
        message : str, optional
            Additional message
        """
        prefix = self._get_step_prefix(step)
        progress_text = f"Progress: {current}/{total}"
        
        if message:
            full_message = f"{prefix} {progress_text} - {message}"
        else:
            full_message = f"{prefix} {progress_text}"
        
        self._print_to_console(full_message)
        if message:
            self._log_to_gui(f"{progress_text} - {message}", log_type='info')
    
    def file_operation(self, step: str, operation: str, file_path: str):
        """
        Log file operations.
        
        Parameters
        ----------
        step : str
            The processing step
        operation : str
            The operation being performed (e.g., 'Loading', 'Saving', 'Processing')
        file_path : str
            The file path
        """
        prefix = self._get_step_prefix(step)
        file_name = os.path.basename(file_path)
        message = f"{operation} file: {file_name}"
        
        full_message = f"{prefix} {message}"
        
        self._print_to_console(full_message)
        self._log_to_gui(message, log_type='file')
    



# Global logger instance
_global_logger = None


def get_logger(log_window=None) -> EEGCometLogger:
    """
    Get the global logger instance.
    
    Parameters
    ----------
    log_window : Optional object
        LogWindow instance for GUI logging
    
    Returns
    -------
    EEGCometLogger
        The logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = EEGCometLogger(log_window)
    elif log_window is not None and _global_logger.log_window is None:
        _global_logger.log_window = log_window
    return _global_logger


def set_log_window(log_window):
    """
    Set the log window for the global logger.
    
    Parameters
    ----------
    log_window : object
        LogWindow instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = EEGCometLogger(log_window)
    else:
        _global_logger.log_window = log_window
