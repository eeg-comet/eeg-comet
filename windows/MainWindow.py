#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:23:14 2021

@author: amin
"""

from PyQt5 import QtGui

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidget
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QComboBox
from PyQt5.QtWidgets import QScrollBar, QRadioButton, QMessageBox, QAction

from windows.PreprocessingWindow import PreprocessingWindow
from functions import find_data

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        self.setGeometry(500, 500, 1000, 1000)
        self.setWindowTitle("Microstate Feature Extraction")
        
        self.MainWindowUI()
        
        self.PreprocessingWindow = PreprocessingWindow()
        
        self.next_button.setDisabled(True)
        self.find_button.setDisabled(True)
        
        self.preprocess_button.setDisabled(True)

        # By Default
        self.enable_raw()
        
        self.load_preprocessed_rb.clicked.connect(self.enable_preprocessed)
        self.load_raw_rb.clicked.connect(self.enable_raw)
        #self.load_preprocessed_rb.clicked.connect(self.disable_others)
        
        
        self.browse_preprocessed_button.clicked.connect(self.browse_preprocessed_files)
        self.browse_raw_button.clicked.connect(self.browse_raw_files)
        
        
        self.find_button.clicked.connect(self.copy_list_files)
        
        
        self.preprocess_button.clicked.connect(self.passingInformation)
        self.preprocess_button.clicked.connect(self.preprocessing_finished)
        
        
        self.dialog = PreprocessingWindow(self)
        
        
        
        # Menu Bar
        quit = QAction("Quit", self)
        quit.triggered.connect(self.close)
        menubar = self.menuBar()
        fmenu = menubar.addMenu("File")
        fmenu.addAction(quit)
        
        
    def MainWindowUI(self):        
        
        # Radio Button - Load Preprocessed or Raw EEG
        self.load_preprocessed_rb = QRadioButton("Load Preprocessed Data", self)
        self.load_preprocessed_rb.move(150, 30)
        self.load_preprocessed_rb.setFixedWidth(500)
        self.load_raw_rb = QRadioButton("Load Raw Data", self)
        self.load_raw_rb.setChecked(True)
        self.load_raw_rb.move(150, 140)
        self.load_raw_rb.setFixedWidth(500)
        
        # Load Preprocessed Data
        self.label = QLabel(self)
        self.label.setText("Select the folder containing preprocessed files")
        self.label.move(150,60)
        self.label.setFixedWidth(400)
        self.browse_preprocessed_button = QPushButton(self)
        self.browse_preprocessed_button.setText("browse")
        self.browse_preprocessed_button.move(150,90)
        self.foldername_preprocessed = QLineEdit(self)
        self.foldername_preprocessed.setReadOnly(True)
        self.foldername_preprocessed.move(250, 90)
        self.foldername_preprocessed.setFixedWidth(400)
        
        # Pattern of Selected Files
        self.label = QLabel(self)
        self.label.setText("Pattern in the name of EEG files")
        self.label.move(150,550)
        self.label.setFixedWidth(300)
        
        self.label = QLabel(self)
        self.label.setText("Starts with:")
        self.label.move(150,580)
        self.label.setFixedWidth(200)
        self.pattern1 = QLineEdit(self)
        self.pattern1.move(250, 580)
        self.pattern1.setFixedWidth(150)
        
        self.label = QLabel(self)
        self.label.setText("Contains:")
        self.label.move(150,610)
        self.label.setFixedWidth(200)
        self.pattern2 = QLineEdit(self)
        self.pattern2.move(250, 610)
        self.pattern2.setFixedWidth(150)
        
        self.label = QLabel(self)
        self.label.setText("Ends with:")
        self.label.move(150,640)
        self.label.setFixedWidth(200)
        self.pattern3 = QLineEdit(self)
        self.pattern3.move(250, 640)
        self.pattern3.setFixedWidth(150)
        
        # List of Selected Files
        self.label = QLabel(self)
        self.label.setText("Selected EEG files")
        self.label.move(150,310)
        self.label.setFixedWidth(250)
        
        self.listfiles = QListWidget(self)
        self.listfiles.move(150, 340)
        self.listfiles.setFixedWidth(500)
        self.listfiles.setFixedHeight(200)
        self.listfiles.setVerticalScrollBar(QScrollBar(self))
        self.listfiles.setAlternatingRowColors(True)
        
        self.n_files = QLabel(self)
        self.n_files.setFont(QtGui.QFont('Times', 20))
        self.n_files.move(150,650)
        self.n_files.setFixedWidth(300)
            
        # Browse Button
        self.label = QLabel(self)
        self.label.setText("Select the folder containing raw EEG files/folders")
        self.label.move(150,170)
        self.label.setFixedWidth(400)
        self.browse_raw_button = QPushButton(self)
        self.browse_raw_button.setText("browse")
        self.browse_raw_button.move(150,200)
        
        self.foldername_raw = QLineEdit(self)
        self.foldername_raw .setReadOnly(True)
        self.foldername_raw .move(250, 200)
        self.foldername_raw .setFixedWidth(400)
        
        # Next Button
        self.next_button = QPushButton(self)
        self.next_button.setText("next")
        self.next_button.move(550,700)
        #self.next_button.clicked.connect(self.clicked)
        
        # DropDown EEG Extensions
        self.label = QLabel(self)
        self.label.setText("Select EEG file extension")
        self.label.move(150,240)
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
        self.comboBox.move(150, 270)
        self.comboBox.setFixedWidth(500)
        #self.comboBox.activated[str].connect(self.copy_list_files)
            
        # Find Files Button
        self.find_button = QPushButton(self)
        self.find_button.setText("find EEGs")
        self.find_button.move(430,580)
        self.find_button.setFixedHeight(90)
        
        # Preprocess Button
        self.preprocess_button = QPushButton(self)
        self.preprocess_button.setText("preprocess\nEEGs")
        self.preprocess_button.move(550,580)
        self.preprocess_button.setFixedHeight(90)
        
        
    # Functions
    
    def enable_raw(self):
        self.browse_preprocessed_button.setDisabled(True)
        self.foldername_preprocessed.setDisabled(True)
        
        self.browse_raw_button.setEnabled(True)
        self.foldername_raw.setEnabled(True)
        
        self.comboBox.setEnabled(True)
        self.listfiles.setEnabled(True)
        self.pattern1.setEnabled(True)
        self.pattern2.setEnabled(True)
        self.pattern3.setEnabled(True)
        
    def enable_preprocessed(self):
        self.browse_preprocessed_button.setEnabled(True)
        self.foldername_preprocessed.setEnabled(True)
        
        self.browse_raw_button.setDisabled(True)
        self.foldername_raw.setDisabled(True)
        self.find_button.setDisabled(True)
        self.preprocess_button.setDisabled(True)
        
        self.comboBox.setDisabled(True)
        self.listfiles.setDisabled(True)
        self.pattern1.setDisabled(True)
        self.pattern2.setDisabled(True)
        self.pattern3.setDisabled(True)
        
        self.foldername_raw.clear()
        self.listfiles.clear()
        self.n_files.clear()
    
    def browse_raw_files(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.foldername_raw.setText(fname)
        if self.foldername_raw.text():
            self.find_button.setEnabled(True)
    
    def browse_preprocessed_files(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.foldername_preprocessed.setText(fname)
        if self.foldername_preprocessed.text():
            self.next_button.setEnabled(True)
    
    def copy_list_files(self):
        
        input_folder = self.foldername_raw.text()
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
        list_eegs = find_data.find_eeg(input_folder, extension, pattern)

        self.listfiles.clear()
        if not list_eegs:
            self.n_files.setText(str("No EEG file found!"))
            self.n_files.setStyleSheet("color: red;")
            self.next_button.setDisabled(True)
        else:
            self.n_files.setText(str(str(len(list_eegs))+" EEG files found"))
            self.n_files.setStyleSheet("color: green;")
            self.preprocess_button.setEnabled(True)
        for i in range(len(list_eegs)):
            self.listfiles.addItem(str(list_eegs[i]))
        
    def copy_list_eegs(self):
        list_eegs = []
        for x in range(self.listfiles.count()):
            list_eegs.append(self.listfiles.item(x).text())
        return list_eegs
        
    def passingInformation(self):
        self.PreprocessingWindow.displayInfo()
        list_eegs = self.copy_list_eegs()
        for i in range(len(list_eegs)):        
            self.PreprocessingWindow.LIST_EEG.addItem(str(list_eegs[i]))
    
    def preprocessing_finished(self):
        self.PreprocessingWindow.signal_end()
        self.preprocess_button.setStyleSheet("background-color : lightgreen; color: black")
        self.preprocess_button.setDisabled(True)
        self.next_button.setEnabled(True)
    
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Quit",
                                               "Are you sure you want to quit?",
                                               QMessageBox.Yes | 
                                               QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()