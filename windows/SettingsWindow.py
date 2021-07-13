#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:42:07 2021

@author: amin
"""

from PyQt5 import QtGui, QtCore

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QButtonGroup
from PyQt5.QtWidgets import QRadioButton, QComboBox


class SettingsWindow(QMainWindow):
    def __init__(self, parent=None):
        super(SettingsWindow, self).__init__(parent)
        
        self.setGeometry(500, 500, 700, 340)
        self.setFixedWidth(700)
        self.setFixedHeight(340)
        self.setWindowTitle("Clustering Settings")

        self.SettingsWindowUI()
        
        self.__smooth_button_clicked = 0
        self.smooth_button.clicked.connect(self.smooth_controller) 
        
        self.savesettings_button.clicked.connect(self.savesettings_controller) 
        
    def SettingsWindowUI(self):  
        
        # Number of Microstate Maps
        self.label = QLabel(self)
        self.label.setText("Number of Microstate Maps")
        self.label.move(75,50)
        self.label.setFixedWidth(400)
        group_radio_nmaps = QButtonGroup(self)
        self.elbow_rb = QRadioButton("Auto", self)
        self.elbow_rb.setToolTip("elbow method to determine the optimal number of clusters")
        self.elbow_rb.move(75, 80)
        self.elbow_rb.setFixedWidth(500)
        group_radio_nmaps.addButton(self.elbow_rb)
        self.user_rb = QRadioButton("User", self)
        self.user_rb.setChecked(True)
        self.user_rb.move(165, 80)
        self.user_rb.setFixedWidth(500)
        group_radio_nmaps.addButton(self.user_rb)
        self.nmaps = QLineEdit(self)
        self.nmaps.setAlignment(QtCore.Qt.AlignCenter)
        self.nmaps.setText('5')
        self.nmaps.setValidator(QtGui.QIntValidator())
        self.nmaps.setMaxLength(2)
        self.nmaps.move(240, 80)
        self.nmaps.setFixedWidth(40)
        
        # Initial centers
        self.label = QLabel(self)
        self.label.setText("Initializer:")
        self.label.setToolTip("Initial centers for clustering")
        self.label.move(360,50)
        self.label.setFixedWidth(200)
        self.randinit_rb = QRadioButton("Random", self)
        self.randinit_rb.setToolTip("random centers from GFP peaks")
        self.randinit_rb.setChecked(True)
        self.randinit_rb.move(440, 50)
        self.randinit_rb.setFixedWidth(500)
        self.kppinit_rb = QRadioButton("K-Means++", self)
        self.kppinit_rb.move(530, 50)
        self.kppinit_rb.setFixedWidth(500)
        
        # Stop Condition
        self.label = QLabel(self)
        self.label.setText("Stop Condition - Tolerance:")
        self.label.setToolTip("if maximum value of change of centers of clusters\nis less than tolerance then algorithm stops processing")
        self.label.move(360,90)
        self.label.setFixedWidth(210)
        self.tol = QLineEdit(self)
        self.tol.setAlignment(QtCore.Qt.AlignCenter)
        tol_validator = QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]+e-[0-9]{,2}"), self.tol)
        self.tol.setValidator(tol_validator)
        self.tol.setText('1e-5')
        self.tol.setMaxLength(5)
        self.tol.move(580, 90)
        self.tol.setFixedWidth(60)
        
        # Smooth Button
        self.label = QLabel(self)
        self.label.setText("Kernel Size:")
        self.label.move(360,130)
        self.label.setFixedWidth(200)
        self.kernel_size = QLineEdit(self)
        self.kernel_size.setAlignment(QtCore.Qt.AlignCenter)
        self.kernel_size.setText('10')
        self.kernel_size.setValidator(QtGui.QIntValidator())
        self.kernel_size.setMaxLength(3)
        self.kernel_size.move(460, 130)
        self.kernel_size.setFixedWidth(40)
        self.smooth_button = QPushButton(self)
        self.smooth_button.setText("smooth GFP")
        self.smooth_button.setToolTip("smoothing using convolution")
        self.smooth_button.move(510,130)
        self.smooth_button.setFixedWidth(130)
        
        # Repeat
        self.label = QLabel(self)
        self.label.setText("Number of Repeats:")
        self.label.move(75,130)
        self.label.setFixedWidth(200)
        self.repeat = QLineEdit(self)
        self.repeat.setAlignment(QtCore.Qt.AlignCenter)
        self.repeat.setText('5')
        self.repeat.setValidator(QtGui.QIntValidator())
        self.repeat.setMaxLength(2)
        self.repeat.move(240, 130)
        self.repeat.setFixedWidth(40)
        
        # Arguments
        self.other1 = QLabel(self)
        self.other1.move(75,170)
        self.other1.setFixedWidth(300)
        self.other2 = QLabel(self)
        self.other2.move(75,200)
        self.other2.setFixedWidth(300)
        self.arg = QComboBox(self)
        '''
        self.arg.addItem("K-Means Distance Metric: Euclidean")
        self.arg.addItem("K-Means Distance Metric: Euclidean Square")
        self.arg.addItem("K-Means Distance Metric: Manhattan")
        self.arg.addItem("K-Means Distance Metric: Chebyshev")
        self.arg.addItem("K-Means Distance Metric: Minkowski")
        self.arg.addItem("X-Means Splitting Criterion: BIC")
        self.arg.addItem("X-Means Splitting Criterion: MNDL")
        '''
        #self.arg.model().item(1).setDisabled(True)
        #self.arg.setCurrentText("K-Means Distance Metric: Euclidean Square")
        self.arg.move(300, 170)
        self.arg.setFixedWidth(340)
        
        # Save Settings Button
        self.savesettings_button = QPushButton(self)
        self.savesettings_button.setText("save settings")
        self.savesettings_button.move(75,250)
        self.savesettings_button.setFixedHeight(50)
        self.savesettings_button.setFixedWidth(565)
        
    
    # Functions
    
    def displaySettings(self):
        self.show()
    
    def smooth_controller(self):
        self.__smooth_button_clicked = 1 - self.__smooth_button_clicked
        if self.__smooth_button_clicked:
            self.smooth_button.setStyleSheet("background-color: lightgreen; color: black")
            self.kernel_size.setDisabled(True)
        else:
            self.smooth_button.setStyleSheet("background-color: None")
            self.kernel_size.setEnabled(True)
        return self.__smooth_button_clicked
        
        
    def savesettings_controller(self):
        self.close()