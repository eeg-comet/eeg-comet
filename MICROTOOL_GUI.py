#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:48:23 2021

Microstate Toolbox GUI

@author: amin
"""

import os
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QPushButton, QStackedWidget
import sys

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidget, QMenu
from PyQt5.QtWidgets import QLabel, QLineEdit, QComboBox, QButtonGroup
from PyQt5.QtWidgets import QScrollBar, QRadioButton, QMessageBox, QAction
from PyQt5.QtCore import QEvent

from PyQt5.QtWidgets import QProgressBar, QCheckBox, QTextEdit

import numpy as np
from functions.concatenate_data import concatenate_files
from functions import extract_features_functions
from functions.clustering_functions import _pre_clustering, initialize_centers, eegInfo
from functions.clustering_functions import clustering_func, clustering_minibatch


from windows.PreprocessingWindow import PreprocessingWindow
from windows.SettingsWindow import SettingsWindow
from windows.MicrostateMapsWindow import MicrostateMapsWindow

from functions import find_data, load_data



class FirstWindow(QMainWindow):
    def __init__(self, parent=None):
        super(FirstWindow, self).__init__(parent)
        
        self.setGeometry(500, 500, 1000, 1000)
        self.setWindowTitle("Microstate Feature Extraction")
        
        self.FirstWindowUI()
        
        self.PreprocessingWindow = PreprocessingWindow()
        self.SecondWindow = SecondWindow()
        
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
        
        
        self.next_button.clicked.connect(self.gotoSecondScreen)
        
        self._filename = ""
        self.plotCHANNELS.triggered.connect(self.plot_CHANNELS)
        self.plotEEG.triggered.connect(self.plot_EEG)
        self.plotPSD.triggered.connect(self.plot_PSD)
        
        # Menu Bar
        quit = QAction("Quit", self)
        quit.triggered.connect(self.close)
        menubar = self.menuBar()
        fmenu = menubar.addMenu("File")
        fmenu.addAction(quit)
        
        
    def FirstWindowUI(self):        
        
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
        self.label.setText("Select the folder containing preprocessed files (.h5)")
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
        self.listfiles.installEventFilter(self)
        
        self.plotCHANNELS = QAction("&Plot Sensors", self)
        self.plotEEG = QAction("&Plot EEG", self)
        self.plotPSD = QAction("&Plot PSD", self)
        
        self.n_files = QLabel(self)
        self.n_files.setFont(QtGui.QFont('Times', 20))
        self.n_files.move(150,680)
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
    
    def get_extension(self):
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
        return extension

    def copy_list_files(self):
        
        input_folder = self.foldername_raw.text()
        pattern="*"
        pattern = self.pattern1.text()+"*"+self.pattern2.text()+"*"+self.pattern3.text()
        extension = self.get_extension()
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
        self.foldername_preprocessed = os.path.join(input_folder, 'output')
        if not os.path.exists(self.foldername_preprocessed):
            os.makedirs(self.foldername_preprocessed)
        
    def copy_list_eegs(self):
        list_eegs = []
        for x in range(self.listfiles.count()):
            list_eegs.append(self.listfiles.item(x).text())
        return list_eegs
    
    
    def eventFilter(self, source, event):
        if event.type()==QEvent.ContextMenu and source is self.listfiles:
            menu = QMenu()
            menu.addAction(self.plotCHANNELS)
            menu.addAction(self.plotEEG)
            menu.addAction(self.plotPSD)
            
            item = source.itemAt(event.pos())
            self._filename = item.text()
            
            menu.exec_(event.globalPos())
                
            return True
        return super().eventFilter(source, event)
    
    
    def plot_CHANNELS(self):
        filename = self._filename
        extension = self.get_extension()
        EEG = load_data.load_eegs(filename, extension)
        EEG.plot_sensors(ch_type='eeg', show_names=True)
    
    def plot_EEG(self):
        filename = self._filename
        extension = self.get_extension()
        EEG = load_data.load_eegs(filename, extension)
        EEG.plot()
        
    def plot_PSD(self):
        filename = self._filename
        extension = self.get_extension()
        EEG = load_data.load_eegs(filename, extension)
        EEG.plot_psd()
    
    
    def passingInformation(self):
        self.PreprocessingWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.PreprocessingWindow.displayInfo()
        self.PreprocessingWindow.foldername_preprocessed.setText(self.foldername_preprocessed)
        extension = self.get_extension()
        self.PreprocessingWindow.eeg_format.setText(extension)
        list_eegs = self.copy_list_eegs()
        for i in range(len(list_eegs)):        
            self.PreprocessingWindow.LIST_EEG.addItem(str(list_eegs[i]))
    
    def preprocessing_finished(self):
        #print("progresss ", self.PreprocessingWindow.__progress)
        #if self.PreprocessingWindow.__processingdone:
        #self.SecondWindow.foldername_preprocessed = self.PreprocessingWindow.foldername_preprocessed.text()
        self.preprocess_button.setStyleSheet("background-color : lightgreen; color: black")
        #self.preprocess_button.setDisabled(True)
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
        
        
        
    def gotoSecondScreen(self):
        secondwindow = SecondWindow()
        widget.addWidget(secondwindow) 
        widget.setCurrentIndex(widget.currentIndex()+1)





class SecondWindow(QMainWindow):
    def __init__(self, parent=None):
        super(SecondWindow, self).__init__(parent)
        
        self.setGeometry(500, 500, 1000, 1000)
        self.setWindowTitle("Microstate Feature Extraction")
        
        self.SecondWindowUI()
        
        self.SettingsWindow = SettingsWindow()
        self.MicrostateDialog = MicrostateMapsWindow()
        
        self._finished_clustering = False
        ### change
        self.foldername_preprocessed = "/media/amin/Seagate Expansion Drive/AMIN/Microstate Toolbox/EEG-Microstate-Feature-Extraction/test_data/output/"
        #self.labeling_button.clicked.connect(self.filter_data)
        
        self.browse_button.clicked.connect(self.browsefiles)
        
        self.settings_button.clicked.connect(self.passingInformation)
        
        self.listoffiles = None
        self.lengthoffiles = None
        self.n_maps = None
        self.micro_labels = None
        self.final_segmentation = None
        
        self.process_button.clicked.connect(self.do_clustering)
        self.process_button.clicked.connect(self.plot_micro)
        
        self.process_button.clicked.connect(self.MicrostateDialog.manual_micro_label)
        
        self.MicrostateDialog.finish_button.clicked.connect(self.pass_labels)
        
        self.extract_button.clicked.connect(self.extract_func)
        self.back_button.clicked.connect(self.gotoFirstScreen)
        
        quit = QAction("Quit", self)
        quit.triggered.connect(self.close)
        
        menubar = self.menuBar()
        fmenu = menubar.addMenu("File")
        fmenu.addAction(quit)
        
        
    def SecondWindowUI(self):        
                
        # DropDown
        self.label = QLabel(self)
        self.label.setText("Clustering Method")
        self.label.move(150,50)
        self.label.setFixedWidth(300)
        
        self.clustering_method = QComboBox(self)
        self.clustering_method.addItem("K-MEANS")
        self.clustering_method.addItem("Mini Batch K-MEANS")
        self.clustering_method.addItem("MODIFIED K-MEANS")
        self.clustering_method.addItem("X-MEANS")
        self.clustering_method.addItem("BSAS")
        self.clustering_method.addItem("CLARANS")
        self.clustering_method.addItem("MBSAS")
        self.clustering_method.addItem("OPTICS")
        self.clustering_method.addItem("ROCK")
        self.clustering_method.move(150, 80)
        self.clustering_method.setFixedWidth(200)
        
        # Advanced Settings Button
        self.settings_button = QPushButton(self)
        self.settings_button.setText("settings")
        self.settings_button.move(350,80)
        self.settings_button.setFixedWidth(300)
        
        # Concatenate
        self.label = QLabel(self)
        self.label.setText("Perform clustering on:")
        self.label.move(150,110)
        self.label.setFixedWidth(200)
        group_radio_data = QButtonGroup(self)
        self.concatenate_rb = QRadioButton("all EEG data concatenated", self)
        self.concatenate_rb.move(350, 110)
        self.concatenate_rb.setFixedWidth(500)
        group_radio_data.addButton(self.concatenate_rb)
        self.separate_rb = QRadioButton("each EEG file separately", self)
        self.separate_rb.setChecked(True)
        self.separate_rb.move(350, 130)
        self.separate_rb.setFixedWidth(500)
        group_radio_data.addButton(self.separate_rb)
        
        # Progress Bar
        self.progress = QProgressBar(self)
        self.progress.move(150,160)
        self.progress.setFixedWidth(500)
        self.progress.setValue(0)
        self.progress.setDisabled(True)
        
        # Process Button
        self.process_button = QPushButton(self)
        self.process_button.setText("start clustering")
        self.process_button.move(150,190)
        self.process_button.setFixedWidth(500)
        
        # Logs
        self.logclustering = QTextEdit(self)
        self.logclustering.move(150, 250)
        self.logclustering.setFixedWidth(500)
        self.logclustering.setFixedHeight(200)
        self.logclustering.setReadOnly(True)
        self.logclustering.setPlainText("Logs\n")
        
        
        '''
        self.plotmaps = QLabel(self)
        #self.plotmaps.setPixmap(QtGui.QPixmap("/home/amin/Encfs/TMSEEG_DATA/microstate_toolbox/maps.png"))
        #self.plotmaps.setPixmap(QtGui.QPixmap("/home/amin/Encfs/TMSEEG_DATA/microstate_toolbox/maps.png"))
        self.plotmaps.move(50,225)
        self.plotmaps.setFixedHeight(200)
        self.plotmaps.setFixedWidth(700)
        '''
        
        
        # Features
        self.label = QLabel(self)
        self.label.setText("Extract features for:")
        self.label.move(150,520)
        self.label.setFixedWidth(200)
        # For each or all
        self.each_cb = QCheckBox("each file", self)
        self.each_cb.setChecked(True)
        self.each_cb.move(305, 520)
        self.each_cb.setFixedWidth(100)
        self.all_cb = QCheckBox("all data", self)
        self.all_cb.setChecked(True)
        self.all_cb.move(400, 520)
        self.all_cb.setFixedWidth(100)
        
        self.label = QLabel(self)
        self.label.setText("Features:")
        self.label.move(150,490)
        self.label.setFixedWidth(200)
        self.foc_cb = QCheckBox("FOC", self)
        self.foc_cb.setToolTip("frequency of occurrence for each microstate map per second")
        self.foc_cb.setChecked(True)
        self.foc_cb.move(305, 490)
        self.foc_cb.setFixedWidth(500)
        self.mmd_cb = QCheckBox("MMD", self)
        self.mmd_cb.setToolTip("mean microstate duration for each microstate map per second")
        self.mmd_cb.setChecked(True)
        self.mmd_cb.move(370, 490)
        self.mmd_cb.setFixedWidth(500)
        self.transition_cb = QCheckBox("Transition Matrix", self)
        self.transition_cb.setChecked(True)
        self.transition_cb.move(440, 490)
        self.transition_cb.setFixedWidth(500)
        
        # Raw segmentation
        self.seg_cb = QCheckBox("save the raw segmentation", self)
        self.seg_cb.setChecked(True)
        self.seg_cb.move(305, 550)
        self.seg_cb.setFixedWidth(500)
        
        
        # Browse for save preprocessed data
        self.label = QLabel(self)
        self.label.setText("Select the folder to save the extracted features")
        self.label.move(150,590)
        self.label.setFixedWidth(400)
        self.browse_button = QPushButton(self)
        self.browse_button.setText("browse")
        self.browse_button.move(150,620)
        self.browse_button.setFixedWidth(110)
        self.foldername_output = QLineEdit(self)
        self.foldername_output.setReadOnly(True)
        self.foldername_output.move(260, 620)
        self.foldername_output.setFixedWidth(390)
        
        
        # Extract button
        self.extract_button = QPushButton(self)
        self.extract_button.setText("extract features")
        self.extract_button.move(150,650)
        self.extract_button.setFixedWidth(500)
        
        # Back Button
        self.back_button = QPushButton(self)
        self.back_button.setText("back")
        self.back_button.move(150,700)
        
        

    # Functions
    
    def displayWindow(self):
        self.show()
    
    def plot_micro(self):
        self.MicrostateDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.MicrostateDialog.show()
        
    def browsefiles(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.foldername_output.setText(fname)
        
            
    def do_clustering(self):
        
        Fs = 250 #from previous window
        #DATA = INPUT_DATA #from previous window
        #print(DATA.shape)
        
        FOLDER = self.foldername_preprocessed
        print(FOLDER)
        # Concatenate data
        if self.concatenate_rb.isChecked()==True:
            DATA, N_CHANNELS, FILENAMES, LENGTH_DATA = concatenate_files(FOLDER)
        
            self.listoffiles = FILENAMES
            self.lengthoffiles = LENGTH_DATA
        
        print(self.SettingsWindow.smooth_controller())
        if self.SettingsWindow.smooth_controller():
            SMOOTHING = int(self.SettingsWindow.kernel_size.text())
        else:
            SMOOTHING = []
        print(SMOOTHING)
        
        if self.SettingsWindow.randinit_rb.isChecked():
                INITIALIZER = "Random"
        elif self.SettingsWindow.kppinit_rb.isChecked():
                INITIALIZER = "K-Means++"
        
        METHOD = self.clustering_method.currentText()
        print(METHOD)
        # modify
        #if self.SettingsWindow.elbow_rb.isChecked():
            #N_STATES = clustering_functions.number_of_clusters(MAPS)
        #elif self.SettingsWindow.user_rb.isChecked():
            #N_STATES = int(self.SettingsWindow.nmaps.text())*2
        N_STATES = int(self.SettingsWindow.nmaps.text())
        print(N_STATES)
        self.n_maps = int(N_STATES/2)
        
        TOLERANCE = float(self.SettingsWindow.tol.text())
        print(TOLERANCE)
        
        if METHOD == "Mini Batch K-MEANS":
                        
            best_maps, final_segmentation, gev = clustering_minibatch(
                                                                FOLDER,
                                                                Fs,
                                                                SMOOTHING,
                                                                N_STATES,
                                                                INITIALIZER,
                                                                TOLERANCE)
            
            
        else:
            
            MAPS, PEAKS = _pre_clustering(DATA, Fs, SMOOTHING)
            
            INITIAL_CENTERS = initialize_centers(
                DATA,                                                 
                MAPS,
                PEAKS,
                N_STATES,
                INITIALIZER)
            print(INITIAL_CENTERS.shape)
        
            METRIC = self.SettingsWindow.arg.currentText()
            print(METRIC)
            
            REPEAT = int(self.SettingsWindow.repeat.text())
            
            best_maps, final_segmentation, gev = clustering_func(
                DATA,
                N_CHANNELS,
                MAPS,
                METHOD,
                N_STATES,
                INITIAL_CENTERS,
                REPEAT,
                TOLERANCE,
                METRIC)
        
        
        self.logclustering.append("Smoothing: "+str(SMOOTHING))
        self.logclustering.append("Clustering Method: "+str(METHOD))
        self.logclustering.append("Number of Maps: "+str(self.n_maps))
        self.logclustering.append("Tolerance: "+str(TOLERANCE))
        self.logclustering.append("Center Initializer: "+str(INITIALIZER))
        #self.logclustering.append("Metric: "+str(METRIC))
        #self.logclustering.append("Number of Repeats: "+str(REPEAT))
        
        
        self.final_segmentation = final_segmentation
        self._finished_clustering = True
        
        self.MicrostateDialog.plot_maps(best_maps, gev, eegInfo(FOLDER))
        
    def pass_labels(self):
        if self.MicrostateDialog.micro_labeled:
            self.micro_labels = self.MicrostateDialog.micro_labels
            self.MicrostateDialog.close()
        
    
    def extract_func(self):
        
        print("Extracting Features ...")
        
        Fs = 250 #from previous window
        Features = []
        '''
        if self.each_cb.isChecked():
            
        if self.all_cb.isChecked():
        '''
        
        Segmentation = self.final_segmentation.tolist()
        Segmentation = list(map(str,Segmentation))
        micro_labels = self.micro_labels
        for i in range(len(micro_labels)):
            Segmentation = np.char.replace(Segmentation, str(i), micro_labels[i])
        print(Segmentation)
        
        if self.foc_cb.isChecked():
            Features.append("FOC")
        if self.mmd_cb.isChecked():
            Features.append("MMD")
        
        extracted_features_df = extract_features_functions.extract_features(
            self.listoffiles, self.lengthoffiles, Segmentation, Fs, Features)
        print(extracted_features_df)
        save_path = self.foldername_output.text()
        # Save Features
        extract_features_functions.save_features(extracted_features_df, save_path)
        print('finished')
    
    def passingInformation(self):
        # Set Defaults
        
        METHOD = self.clustering_method.currentText()
        if METHOD == "K-MEANS":
            self.SettingsWindow.other1.setText("K-Means Distance Metric:")
            self.SettingsWindow.arg.clear()
            self.SettingsWindow.arg.addItem("Euclidean")
            self.SettingsWindow.arg.addItem("Euclidean Square")
            self.SettingsWindow.arg.addItem("Manhattan")
            self.SettingsWindow.arg.addItem("Chebyshev")
            self.SettingsWindow.arg.addItem("Minkowski")
            self.SettingsWindow.arg.setCurrentText("Euclidean Square")
        #elif METHOD == "MINI BATCH K-MEANS":
        #    
        elif METHOD == "X-MEANS":
            self.SettingsWindow.other1.setText("X-Means Splitting Criterion:")
            self.SettingsWindow.arg.clear()
            self.SettingsWindow.arg.addItem("Bayesian Information Criterion")
            self.SettingsWindow.arg.addItem("Minimum Noiseless Description Length")
            self.SettingsWindow.arg.setCurrentText("Bayesian Information Criterion")
        
        
        self.SettingsWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.SettingsWindow.show()
        
    def gotoFirstScreen(self):
        firstwindow = FirstWindow()
        widget.addWidget(firstwindow) 
        widget.setCurrentIndex(widget.currentIndex()+1)
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Quit",
                                               "Are you sure you want to quit?",
                                               QMessageBox.Yes | 
                                               QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        



app = QApplication(sys.argv)
widget = QStackedWidget()
firstwindow = FirstWindow()
widget.addWidget(firstwindow) 
widget.setFixedWidth(800)
widget.setFixedHeight(800)

widget.show()
sys.exit(app.exec_())
