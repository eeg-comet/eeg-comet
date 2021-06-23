#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:48:23 2021

Microstate Toolbox GUI

@author: amin
"""

from PyQt5 import QtGui

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QMessageBox, QComboBox
from PyQt5.QtWidgets import QScrollBar, QStackedWidget


import sys
import os
from fnmatch import fnmatch


class MyWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        
        self.setGeometry(500, 500, 1000, 1000)
        self.setWindowTitle("Microstate Feature Extraction")
        
        self.MainWindowUI()
        
        self.next.setDisabled(True)
        self.find_button.setDisabled(True)
        self.browse_button.clicked.connect(self.browsefiles)
        self.find_button.clicked.connect(self.copy_list_files)
        
        '''
        # Menu Bar
        quit = QAction("Quit", self)
        quit.triggered.connect(self.close)
        menubar = self.menuBar()
        fmenu = menubar.addMenu("File")
        fmenu.addAction(quit)
        '''
        
    def MainWindowUI(self):        
        
        
        # Pattern of Selected Files
        self.label = QLabel(self)
        self.label.setText("Pattern in the name of EEG files")
        self.label.move(150,500)
        self.label.setFixedWidth(300)
        
        self.label = QLabel(self)
        self.label.setText("Starts with:")
        self.label.move(150,530)
        self.label.setFixedWidth(200)
        self.pattern1 = QLineEdit(self)
        self.pattern1.move(250, 530)
        self.pattern1.setFixedWidth(150)
        
        
        self.label = QLabel(self)
        self.label.setText("Contains:")
        self.label.move(150,560)
        self.label.setFixedWidth(200)
        self.pattern2 = QLineEdit(self)
        self.pattern2.move(250, 560)
        self.pattern2.setFixedWidth(150)
        
        
        self.label = QLabel(self)
        self.label.setText("Ends with:")
        self.label.move(150,590)
        self.label.setFixedWidth(200)
        self.pattern3 = QLineEdit(self)
        self.pattern3.move(250, 590)
        self.pattern3.setFixedWidth(150)
        
        
        # List of Selected Files
        self.label = QLabel(self)
        self.label.setText("Selected EEG files")
        self.label.move(150,200)
        self.label.setFixedWidth(200)
        
        self.listfiles = QListWidget(self)
        self.listfiles.move(150, 230)
        self.listfiles.setFixedWidth(500)
        self.listfiles.setFixedHeight(250)
        self.listfiles.setVerticalScrollBar(QScrollBar(self))
        self.listfiles.setAlternatingRowColors(True)
        
        
        self.n_files = QLabel(self)
        self.n_files.setFont(QtGui.QFont('Times', 20))
        self.n_files.move(150,650)
        self.n_files.setFixedWidth(300)
            
        # Browse Button
        self.label = QLabel(self)
        self.label.setText("Select the folder containing EEG files/folders")
        self.label.move(150,50)
        self.label.setFixedWidth(400)
        self.browse_button = QPushButton(self)
        self.browse_button.setText("browse")
        self.browse_button.move(550,50)
        
        self.foldername = QLineEdit(self)
        self.foldername.setReadOnly(True)
        self.foldername.move(150, 90)
        self.foldername.setFixedWidth(500)
        
        
        # Next Button
        self.next = QPushButton(self)
        self.next.setText("next")
        self.next.move(550,700)
        #self.next.clicked.connect(self.clicked)
        
        # DropDown EEG Extensions
        self.label = QLabel(self)
        self.label.setText("Select EEG file extension")
        self.label.move(150,130)
        self.label.setFixedWidth(300)
                
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("BrainVision (.vhdr, .vmrk, .eeg)")
        self.comboBox.addItem("European data format (.edf)")
        self.comboBox.addItem("BioSemi data format (.bdf)")
        self.comboBox.addItem("General data format (.gdf)")
        self.comboBox.addItem("Neuroscan CNT (.cnt)")
        self.comboBox.addItem("EGI simple binary (.egi)")
        self.comboBox.addItem("EGI MFF (.mff)")
        self.comboBox.addItem("EEGLAB files (.set, .fdt)")
        self.comboBox.addItem("Nicolet (.data)")
        self.comboBox.addItem("eXimia EEG data (.nxe)")
        self.comboBox.addItem("Persyst EEG data (.lay, .dat)")
        self.comboBox.addItem("Nihon Kohden EEG data (.eeg, .21e, .pnt, .log)")
        self.comboBox.move(150, 160)
        self.comboBox.setFixedWidth(500)
        #self.comboBox.activated[str].connect(self.copy_list_files)
            
        
        # Find Files Button
        self.find_button = QPushButton(self)
        self.find_button.setText("find EEGs")
        self.find_button.move(550,530)
        self.find_button.setFixedHeight(90)
        
    # Functions
    def browsefiles(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.foldername.setText(fname)
        if self.foldername.text():
            self.find_button.setEnabled(True)
                    
    def copy_list_files(self, text):
        
        mylist = []
        #input_folder="/home/amin/Encfs/TMSEEG_DATA/microstate_toolbox/data/"
        input_folder = self.foldername.text()
        pattern="*"
        pattern = self.pattern1.text()+"*"+self.pattern2.text()+"*"+self.pattern3.text()
        
        selected_extension = self.comboBox.currentText()
        if selected_extension == "BrainVision (.vhdr, .vmrk, .eeg)":
            extension=".vhdr"
        elif selected_extension == "European data format (.edf)":
            extension=".edf"
        elif selected_extension == "BioSemi data format (.bdf)":
            extension=".bdf"
        elif selected_extension == "General data format (.gdf)":
            extension=".gdf"
        elif selected_extension == "Neuroscan CNT (.cnt)":
            extension=".cnt"
        elif selected_extension == "EGI simple binary (.egi)":
            extension=".egi"
        elif selected_extension == "EGI MFF (.mff)":
            extension=".mff"
        elif selected_extension == "EEGLAB files (.set, .fdt)":
            extension=".set"
        elif selected_extension == "Nicolet (.data)":
            extension=".data"
        elif selected_extension == "eXimia EEG data (.nxe)":
            extension=".nxe"
        elif selected_extension == "Persyst EEG data (.lay, .dat)":
            extension=".lay"
        elif selected_extension == "Nihon Kohden EEG data (.eeg, .21e, .pnt, .log)":
            extension=".eeg"
            
        for path, subdirs, files in os.walk(input_folder):
            for name in files:
                if fnmatch(name, pattern+extension):
                    mylist.append(os.path.join(path, name))
                    
        self.listfiles.clear()
        if not mylist:
            self.n_files.setText(str("No EEG file found!"))
            self.n_files.setStyleSheet("color: red;")
            self.next.setDisabled(True)
        else:
            self.n_files.setText(str(str(len(mylist))+" EEG files found"))
            self.n_files.setStyleSheet("color: green;")
            self.next.setEnabled(True)
        for i in range(len(mylist)):
            self.listfiles.addItem(str(mylist[i]))
    
    
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Quit",
                                               "Are you sure you want to quit?",
                                               QMessageBox.Yes | 
                                               QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    '''



def window():
    app = QApplication(sys.argv)
    mainwindow = MyWindow()
    widget = QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(800)
    widget.setFixedHeight(800)
    
    widget.show()
    sys.exit(app.exec_())
    
window()