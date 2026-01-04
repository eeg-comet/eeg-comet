"""EEG-COMET GUI application entrypoint."""

import os
import sys
import warnings
import logging

import mne
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from controllers.main_microstate_window import MainMicrostateWindow
from gui_utils.terminal_logger import get_logger

# Silence TensorFlow warnings before any imports that might use it
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide INFO and WARNING messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.*")

# Suppress MNE warnings about electrode positions globally
mne.set_log_level('ERROR')

# Also suppress via Python's logging system
class MNEWarningFilter(logging.Filter):
    """Filter to suppress specific MNE warnings about electrode positions."""
    def filter(self, record):
        # Filter out electrode position warnings
        try:
            if hasattr(record, 'getMessage'):
                message = record.getMessage()
                if "Did not find any electrode locations" in message:
                    return False
                if "digitization points do not correspond" in message:
                    return False
            # Also check record.msg directly
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                if "Did not find any electrode locations" in record.msg:
                    return False
                if "digitization points do not correspond" in record.msg:
                    return False
        except Exception:
            # If anything goes wrong with filtering, allow the message through
            pass
        return True

# Apply filter to root logger to catch all MNE warnings
logging.getLogger().addFilter(MNEWarningFilter())

# Initialize global logger
logger = get_logger()


class CustomApplicationContext:
    """Custom application context for EEG-COMET."""

    def __init__(self):
        """Create the QApplication and resolve base path for resources."""
        self.app = QApplication([])
        self.base_path = os.path.dirname(__file__)

    def get_resource(self, path):
        """Return the full path of a resource file."""
        return os.path.join(self.base_path, "ui", path)


def run_application():
    """Run the EEG-COMET application."""
    # Display welcome message
    logger.toolbox_header("EEG-COMET", "(EEG Comprehensive Microstate Extraction Toolbox)")
    logger.processing_info("ORGANIZATION", "SFU eBrain Lab")
    logger.processing_info("CONTACT", "     https://www.ebrainlab.ca/about-us")
    logger.processing_info("GITHUB", "      https://github.com/eBrainLab/eeg-comet")
    logger.processing_info("MAINTENANCE", " https://github.com/aminkabir")

    try:
        # Create a custom application context
        app_context = CustomApplicationContext()
        app = app_context.app

        # Initialize main window
        window = MainMicrostateWindow(app_context, parent=None)

        # Set the application icon
        icon_path = app_context.get_resource("eeg_comet_logo.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        else:
            logger.warning("STARTUP", "Application icon not found")

        # Show the window
        window.show()
        logger.processing_success("STARTUP", "     EEG-COMET GUI Started Successfully")

        # Start the application
        exit_code = app.exec_()

        # Cleanup and close all windows
        # Add final closing message to log window if available
        if (
            hasattr(window, "comet")
            and hasattr(window.comet, "LogWindow")
            and window.comet.LogWindow
        ):
            window.comet.LogWindow.append_log("EEG-COMET Session Ended", log_type="section")
            window.comet.LogWindow.append_log("Thank you for using EEG-COMET!", log_type="info")

        # Close the main window (this will trigger closeEvent and close all dialogs)
        if hasattr(window, "close"):
            window.close()

        # Force close any remaining windows
        for widget in app.topLevelWidgets():
            if widget.isVisible():
                widget.close()

        # Add separator before shutdown message
        print()  # Empty line
        print("=" * 60)  # Separator line

        logger.processing_info("SHUTDOWN", "    EEG-COMET GUI Closed Successfully")
        sys.exit(exit_code)

    except Exception as e:
        logger.error("STARTUP", f"Failed to Start EEG-COMET GUI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_application()
