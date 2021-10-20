import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow


class MainMicrostateWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainMicrostateWindow, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = self.__class__.__name__.lower()
        uifile = os.path.join(basepath, 'designer/%s.ui' % basename)
        self.ui = uic.loadUi(uifile, self)
