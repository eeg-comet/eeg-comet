"""
MicrostateTools Main

"""
# define authorship information
__authors__ = ['Amin Kabir', 'Raaj Chatterjee']
__author__ = ','.join(__authors__)
__credits__ = []
__copyright__ = 'Copyright (c) 2021'
__license__ = 'GPL'

# maintanence information
__maintainer__ = 'Amin Kabir'
__email__ = 'kabir@sfu.ca'

# define version information
__requires__ = ['PyQt5']
__version_info__ = (0, 0, 0)
__version__ = 'v%i.%02i.%02i' % __version_info__
__revision__ = __version__

import os.path
import sys
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QApplication
from gui.mainmicrostatewindow import MainMicrostateWindow

if __name__ == '__main__':

    # 1. Instantiate ApplicationContext
    appctxt = ApplicationContext()
    app = None
    if not QApplication.instance():
        app = QApplication(sys.argv)

    window = MainMicrostateWindow(appctxt)
    window.show()

    # 2. Invoke appctxt.app.exec_()
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)
