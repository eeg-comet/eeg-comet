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


from gui_utils.logger import get_logger

# Initialize global logger
logger = get_logger()


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
    # Display welcome message
    logger.toolbox_header("EEG-COMET", "(EEG Comprehensive Microstate Extraction Toolbox)")
    logger.processing_info("INITIALIZATION", "Organization: SFU eBrain Lab (https://www.ebrainlab.ca/)")
    logger.processing_info("INITIALIZATION", "GitHub: https://github.com/eBrainLab/eeg-comet/")
    logger.processing_info("INITIALIZATION", "Contacting authors: Amin Kabir, Faranak Farzan")
    
    try:
        # Create a custom application context
        logger.processing_start("INITIALIZATION", "Initializing graphical user interface")
        app_context = CustomApplicationContext()
        app = app_context.app

        # Initialize main window
        window = MainMicrostateWindow(app_context, parent=None)
        
        # Set the application icon
        icon_path = app_context.get_resource("eeg_comet_logo.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        else:
            logger.warning("INITIALIZATION", "Application icon not found")

        # Show the window
        window.show()
        logger.processing_success("INITIALIZATION", "Application started successfully")
        
        # Start the application
        exit_code = app.exec_()
        
        # Cleanup
        logger.processing_info("INITIALIZATION", "Application closed")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error("INITIALIZATION", f"Failed to start application: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    run_application()
