import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QWizard

class MainWindow(QMainWindow):
    """ Main Window class for the Nexsys filesystem application. """
    switch_window = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = self.__class__.__name__.lower()
        uifile = os.path.join(basepath, 'designer/%s.ui' % basename)
        self.ui = uic.loadUi(uifile, self)

        dialog = LoadDialog()
        something = self.dialog.getFiles
        # Connections
        self.ui.pushButton.clicked.connect(self.launchWizard)

        # State Update Functions
        def displayController(self):

        #
        def loadSettingsFromFile(self):

    # def launchWizard(self):
    #     from microstate_tool.gui.firststep import FirstStep
    #     """ Launches the export movies wizard. """
    #     wizard = QWizard(self)
    #     wizard.addPage(FirstStep(wizard))
    #     wizard.exec_()


