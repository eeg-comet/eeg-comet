#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:19:58 2021

@author: amin
"""

from PyQt5 import QtGui, QtCore

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QMessageBox, QComboBox
from PyQt5.QtWidgets import QScrollBar, QStackedWidget, QRadioButton, QProgressBar

from functions import find_data, preprocess_functions


class PreprocessingWindow(QMainWindow):
    def __init__(self, parent=None):
        super(PreprocessingWindow, self).__init__(parent)
        
        self.setGeometry(500, 500, 700, 550)
        self.setFixedWidth(700)
        self.setFixedHeight(550)
        self.setWindowTitle("Preprocessing")
        
        self.PreprocessingWindowUI()
        
        self.browse_button.setDisabled(True)
        self.foldername.setDisabled(True)
        self.preprocess_button.setDisabled(True)
        
        self.__filter_true = 0
        self.__downsample_true = 0
        self.__smooth_button_clicked = 0
        self.__finished = 0
        
        self.filter_button.clicked.connect(self.filter_data)
        self.downsample_button.clicked.connect(self.downsample)
        self.reset.clicked.connect(self.reset_func)
        
        self.browse_button.clicked.connect(self.browsefiles)     
        
        self.preprocess_button.clicked.connect(self.preprocess_func)
        
        
    def PreprocessingWindowUI(self):        
        
        # List of EEGs
        self.label = QLabel(self)
        self.label.setText("List of EEG files")
        self.label.move(100,20)
        self.label.setFixedWidth(300)
        self.LIST_EEG = QListWidget(self)
        self.LIST_EEG.move(100,50)
        self.LIST_EEG.setFixedWidth(500)
        self.LIST_EEG.setFixedHeight(150)
        
        # Filter Button
        self.label = QLabel(self)
        self.label.setText("Bandpass Filter Data")
        self.label.move(100,205)
        self.label.setFixedWidth(400)
        self.filter_button = QPushButton(self)
        self.filter_button.setText("filter")
        self.filter_button.move(100,335)
        self.filter_button.setFixedWidth(155)
        
        # Lowcut
        self.label = QLabel(self)
        self.label.setText("lowcut")
        self.label.move(110,235)
        self.label.setFixedWidth(200)
        self.lowcut = QLineEdit(self)
        self.lowcut.setAlignment(QtCore.Qt.AlignCenter)
        self.lowcut.setText('2')
        self.lowcut.setValidator(QtGui.QIntValidator())
        self.lowcut.setMaxLength(2)
        self.lowcut.move(180, 235)
        self.lowcut.setFixedWidth(40)
        self.label = QLabel(self)
        self.label.setText("Hz")
        self.label.move(225,235)
        
        # Highcut
        self.label = QLabel(self)
        self.label.setText("highcut")
        self.label.move(110,265)
        self.label.setFixedWidth(200)
        self.highcut = QLineEdit(self)
        self.highcut.setAlignment(QtCore.Qt.AlignCenter)
        self.highcut.setText('20')
        self.highcut.setValidator(QtGui.QIntValidator())
        self.highcut.setMaxLength(2)
        self.highcut.move(180, 265)
        self.highcut.setFixedWidth(40)
        self.label = QLabel(self)
        self.label.setText("Hz")
        self.label.move(225,265)
        
        # Order
        self.label = QLabel(self)
        self.label.setText("order")
        self.label.move(110,295)
        self.label.setFixedWidth(200)
        self.order = QLineEdit(self)
        self.order.setAlignment(QtCore.Qt.AlignCenter)
        self.order.setText('5')
        self.order.setValidator(QtGui.QIntValidator())
        self.order.setMaxLength(2)
        self.order.move(180, 295)
        self.order.setFixedWidth(40)
        
        
        # Downsample Button
        self.label = QLabel(self)
        self.label.setText("Downsample Data")
        self.label.move(460,205)
        self.label.setFixedWidth(400)
        
        self.label = QLabel(self)
        self.label.setText("fs")
        self.label.move(475,265)
        self.label.setFixedWidth(200)
        self.sample_rate = QLineEdit(self)
        self.sample_rate.setAlignment(QtCore.Qt.AlignCenter)
        self.sample_rate.setText('250')
        self.sample_rate.setValidator(QtGui.QIntValidator())
        self.sample_rate.setMaxLength(4)
        self.sample_rate.move(500, 265)
        self.sample_rate.setFixedWidth(60)
        
        self.label = QLabel(self)
        self.label.setText("Hz")
        self.label.move(565,265)
        
        
        self.downsample_button = QPushButton(self)
        self.downsample_button.setText("downsample")
        self.downsample_button.move(460,335)
        self.downsample_button.setFixedWidth(140)
        
        # Reset Button
        self.reset = QPushButton(self)
        self.reset.setText("reset")
        self.reset.move(310,335)
        
        # Browse for save preprocessed data
        self.label = QLabel(self)
        self.label.setText("Select the folder to save the preprocessed EEG files")
        self.label.move(100,385)
        self.label.setFixedWidth(400)
        self.browse_button = QPushButton(self)
        self.browse_button.setText("save path")
        self.browse_button.move(100,415)
        
        self.foldername = QLineEdit(self)
        self.foldername.setReadOnly(True)
        self.foldername.move(210, 415)
        self.foldername.setFixedWidth(390)
        
        # Progress Bar
        self.progress = QProgressBar(self)
        self.progress.move(100,455)
        self.progress.setFixedWidth(500)
        self.progress.setValue(0)
        self.progress.setDisabled(True)
        
        # Preprocess button
        self.preprocess_button = QPushButton(self)
        self.preprocess_button.setText("preprocess all data")
        self.preprocess_button.move(100,485)
        self.preprocess_button.setFixedWidth(500)
        
        
    # Functions
    
    def displayInfo(self):
        self.show()
    
    def filter_data(self, event):
        
        if self.lowcut.text() >= self.highcut.text():
            reply = QMessageBox.question(self, "no", QMessageBox.OK |
                                                  QMessageBox.Ignore)
            if reply == QMessageBox.OK:
                event.accept()
        else:
            self.lowcut.setDisabled(True)
            self.highcut.setDisabled(True)
            self.order.setDisabled(True)
            self.filter_button.setDisabled(True)
            self.filter_button.setStyleSheet("background-color : lightgreen; color: black")
            
            self.browse_button.setEnabled(True)
            self.foldername.setEnabled(True)
            if self.foldername.text():
                self.preprocess_button.setEnabled(True)
                self.__filter_true = 1
    
    def downsample(self):
        self.downsample_button.setStyleSheet("background-color : lightgreen; color: black")
        self.sample_rate.setDisabled(True)
        self.downsample_button.setDisabled(True)
        
        self.browse_button.setEnabled(True)
        self.foldername.setEnabled(True)
        if self.foldername.text():
            self.preprocess_button.setEnabled(True)
            self.__downsample_true = 1
    
    def preprocess_func(self):
        
        self.lowcut.setDisabled(True)
        self.highcut.setDisabled(True)
        self.order.setDisabled(True)
        self.filter_button.setDisabled(True)
        self.sample_rate.setDisabled(True)
        self.downsample_button.setDisabled(True)
        self.browse_button.setDisabled(True)
        
        list_eegs = []
        for x in range(self.LIST_EEG.count()):
            list_eegs.append(self.LIST_EEG.item(x).text())
        
        channels2remove = preprocess_functions.chan2rm_eeg(list_eegs)
        
        for filename in list_eegs:
            DATA = preprocess_functions.preprocess_eeg(filename, channels2remove,
                                self.__filter_true, self.lowcut.text(),
                                self.highcut.text(), self.order.text(),
                                self.__downsample_true, self.sample_rate.text())
            
            preprocess_functions.save_preprocessed_eeg(DATA, filename, self.foldername.text())
            self.progress.setValue(100*(list_eegs.index(filename)+1)/len(list_eegs))
        self.__finished = 1
        
    def signal_end(self):
        if self.__finished:
            self.preprocess_button.setStyleSheet("background-color : lightgreen; color: black")
            self.preprocess_button.setDisabled(True)
            self.close()
    
    def reset_func(self):
        self.lowcut.setEnabled(True)
        self.highcut.setEnabled(True)
        self.order.setEnabled(True)
        self.sample_rate.setEnabled(True)
        
        self.filter_button.setEnabled(True)
        self.filter_button.setStyleSheet("background-color : None")
        self.downsample_button.setEnabled(True)
        self.downsample_button.setStyleSheet("background-color : None")
        
        self.browse_button.setDisabled(True)
        self.foldername.setDisabled(True)
        self.preprocess_button.setDisabled(True)
        
        self.progress.setValue(0)
        self.progress.setDisabled(True)
        self.preprocess_button.setStyleSheet("background-color : None")
        
    
    def browsefiles(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.foldername.setText(fname)
        
        if self.foldername.text():
            self.preprocess_button.setEnabled(True)
            self.progress.setEnabled(True)