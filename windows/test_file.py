# import PyQt5.uic
# from PyQt5 import QtGui, QtCore
#
# from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QLineEdit
# from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
#
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure
#
# import numpy as np
# import mne
# from matplotlib import pyplot as plt
# from functions.test_classifier import label_micromap

import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.ui = uic.loadUi("../designer/FirstWindow.ui", self)


if __name__ == '__main__':
    app = None
    if not app:
        app = QtGui.QApplication([])


app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("../designer/FirstWindow.ui")
window.show()
app.exec()
