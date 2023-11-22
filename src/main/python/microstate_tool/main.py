"""
MicrostateTools Main
"""

# Define authorship information
__authors__ = ['Amin Kabir', 'Raaj Chatterjee']
__author__ = ','.join(__authors__)
__credits__ = []
__copyright__ = 'Copyright (c) 2021'
__license__ = 'GPL'

# Maintenance information
__maintainer__ = 'Amin Kabir'
__email__ = 'kabir@sfu.ca'

# Define version information
__requires__ = ['PyQt5']
__version_info__ = (0, 0, 0)
__version__ = f'v{__version_info__[0]}.{__version_info__[1]:02d}.{__version_info__[2]:02d}'
__revision__ = __version__

import sys
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QApplication
from gui.mainmicrostatewindow import MainMicrostateWindow

def run_application():
    """
    Run the MicrostateTools application.
    """
    # Instantiate ApplicationContext
    appctxt = ApplicationContext()

    # Check if QApplication instance already exists
    app = QApplication.instance() or QApplication(sys.argv)

    window = MainMicrostateWindow(appctxt)
    window.show()

    # Invoke appctxt.app.exec_()
    exit_code = appctxt.app.exec_()

    sys.exit(exit_code)

if __name__ == '__main__':
    run_application()
