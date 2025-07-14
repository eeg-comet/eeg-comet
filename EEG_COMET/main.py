"""
EEG-COMET Main Application

Authors:
    - Amin Kabir
    - Raaj Chatterjee
    - Faranak Farzan

Organization:
    SFU eBrain Lab
    Website: www.ebrainlab.ca
"""

import sys
import os
import warnings
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from controllers.main_microstate_window import MainMicrostateWindow

# Silence TensorFlow warnings before any imports that might use it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
warnings.filterwarnings('ignore', category=UserWarning, module='.*tensorflow.*')


class TerminalLogger:
    """
    Simple terminal logger for important steps only.
    """
    @staticmethod
    def info(message):
        """Print an info message with prefix."""
        print(f"[INFO] {message}")
    
    @staticmethod
    def success(message):
        """Print a success message with checkmark."""
        print(f"✓ {message}")
    
    @staticmethod
    def error(message):
        """Print an error message with prefix."""
        print(f"[ERROR] {message}")
    
    @staticmethod
    def warning(message):
        """Print a warning message with prefix."""
        print(f"[WARNING] {message}")
    
    @staticmethod
    def header(message):
        """Print a header message with separators."""
        print("\n" + "=" * 60)
        print(f"🧠 {message}")
        print("=" * 60)


class CustomApplicationContext:
    """
    Custom application context for EEG-COMET.
    """
    def __init__(self):
        self.app = QApplication([])
        self.base_path = os.path.dirname(__file__)

    def get_resource(self, path):
        """
        Get the full path of a resource file.
        """
        return os.path.join(self.base_path, "ui", path)


def run_application():
    """
    Run the EEG-COMET application.
    """
    logger = TerminalLogger()
    
    # Display welcome message
    logger.header("EEG-COMET (EEG Comprehensive Microstate Extraction Toolbox)")
    logger.info("Authors: Amin Kabir, Raaj Chatterjee, Faranak Farzan")
    logger.info("Organization: SFU eBrain Lab (www.ebrainlab.ca)")
    
    try:
        # Create a custom application context
        logger.info("Initializing application...")
        app_context = CustomApplicationContext()
        app = app_context.app

        # Initialize main window
        logger.info("Loading main window...")
        window = MainMicrostateWindow(app_context, parent=None)
        
        # Set the application icon
        icon_path = app_context.get_resource("eeg_comet_logo.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        else:
            logger.warning("Application icon not found")

        # Show the window
        window.show()
        logger.success("Application started successfully!")
        
        # Start the application
        exit_code = app.exec_()
        
        # Cleanup
        logger.info("Application closed")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    run_application()
