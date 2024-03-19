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
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from controllers.mainmicrostatewindow import MainMicrostateWindow


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
    # Create a custom application context
    app_context = CustomApplicationContext()

    # Check if QApplication instance already exists
    app = app_context.app

    # Use app_context in MainMicrostateWindow or other windows
    window = MainMicrostateWindow(app_context, parent=None)
    window.show()

    # Set the application icon
    icon_path = app_context.get_resource("eeg_comet_logo.png")
    app.setWindowIcon(QIcon(icon_path))

    # Invoke app.exec_()
    exit_code = app.exec_()

    # Cleanup or perform any necessary actions
    sys.exit(exit_code)


if __name__ == '__main__':
    run_application()
