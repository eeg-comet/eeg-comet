"""
AminTools Main

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

import os
import numpy as np
import sys

from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtWidgets import QApplication, QPushButton, QStackedWidget
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidget, QMenu
from PyQt5.QtWidgets import QLabel, QLineEdit, QComboBox, QButtonGroup
from PyQt5.QtWidgets import QScrollBar, QRadioButton, QMessageBox, QAction
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QProgressBar, QCheckBox, QTextEdit


def main(argv=None):
    app = None
    if not QApplication.instance():
        app = QApplication(sys.argv)

    from microstate_tool.gui.mainwindow import MainWindow
    window = MainWindow()
    window.show()

    if app:
        return app.exec_()

    return 0


if __name__ == '__main__':
    from functions.concatenate_data import concatenate_files
    from functions import extract_features_functions
    from functions.clustering_functions import _pre_clustering, initialize_centers, eegInfo
    from functions.clustering_functions import clustering_func, clustering_minibatch
    from functions import find_data, load_data

    sys.exit(main(sys.argv))
