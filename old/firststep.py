import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QWizardPage

class FirstStep(QWizardPage):
    """ Main Window class for the Nexsys filesystem application. """

    def __init__(self, parent):
        super(FirstStep, self).__init__(parent)
        self.setTitle('My First Step')
        self.setSubTitle('Setup movie specific data')

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = self.__class__.__name__.lower()
        uifile = os.path.join(basepath, 'designer/%s.ui' % basename)
        self.ui = uic.loadUi(uifile, self)
