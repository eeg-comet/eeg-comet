#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:46:03 2021

@author: amin
"""

from PyQt5 import QtCore

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QProgressBar
from PyQt5.QtWidgets import QAction, QLabel, QLineEdit, QPushButton, QButtonGroup
from PyQt5.QtWidgets import QCheckBox, QRadioButton, QComboBox, QMessageBox

from windows.SettingsWindow import SettingsWindow
from windows.MicrostateMapsWindow import MicrostateMapsWindow

import numpy as np
from functions import extract_features_functions
from functions import clustering_functions

class ClusteringWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ClusteringWindow, self).__init__(parent)
        
        self.setGeometry(500, 500, 1000, 1000)
        self.setWindowTitle("Microstate Feature Extraction")
        
        self.ClusteringWindowUI()
        
        self.SettingsWindow = SettingsWindow()
        self.MicrostateDialog = MicrostateMapsWindow()
        
        self._finished_clustering = 0
        
        #self.labeling_button.clicked.connect(self.filter_data)
        
        self.browse_button.clicked.connect(self.browsefiles)
        
        self.settings_button.clicked.connect(self.passingInformation)
        
        #self.micro_labels = None
        self.n_maps = None
        self.final_segmentation = None
        
        self.process_button.clicked.connect(self.do_clustering)
        self.process_button.clicked.connect(self.plot_micro)
        
        self.process_button.clicked.connect(self.MicrostateDialog.manual_micro_label)
        
        self.MicrostateDialog.finish_button.clicked.connect(self.pass_labels)
        
        self.extract_button.clicked.connect(self.extract_func)
        
        quit = QAction("Quit", self)
        quit.triggered.connect(self.close)
        
        menubar = self.menuBar()
        fmenu = menubar.addMenu("File")
        fmenu.addAction(quit)
        
        
    def ClusteringWindowUI(self):        
                
        # DropDown
        self.label = QLabel(self)
        self.label.setText("Clustering Method")
        self.label.move(150,50)
        self.label.setFixedWidth(300)
        
        self.clustering_method = QComboBox(self)
        self.clustering_method.addItem("K-MEANS")
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
        
        # Plot Microstate Maps
        self.label = QLabel(self)
        self.label.setText("Labeling Microstate Maps")
        self.label.move(150,225)
        self.label.setFixedWidth(400)
        
        self.micromaps = QLabel(self)
        self.micromaps.move(50,225)
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
        self.foldername = QLineEdit(self)
        self.foldername.setReadOnly(True)
        self.foldername.move(260, 620)
        self.foldername.setFixedWidth(390)
        
        # Extract button
        self.extract_button = QPushButton(self)
        self.extract_button.setText("extract features")
        self.extract_button.move(150,650)
        self.extract_button.setFixedWidth(500)
        
        # Back Button
        self.back = QPushButton(self)
        self.back.setText("back")
        self.back.move(150,700)
        
        

    # Functions
    
    def plot_micro(self):
        self.MicrostateDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.MicrostateDialog.show()
        
    def browsefiles(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.foldername.setText(fname)
        
            
    def do_clustering(self):
        
        Fs = 250 #from previous window
        DATA = clustering_functions.INPUT_DATA #from previous window
        print(self.SettingsWindow.smooth_controller())
        if self.SettingsWindow.smooth_controller():
            SMOOTHING = int(self.SettingsWindow.kernel_size.text())
        else:
            SMOOTHING = []
        print(SMOOTHING)
        MAPS, PEAKS = clustering_functions._pre_clustering(DATA, Fs, SMOOTHING)
        METHOD = self.clustering_method.currentText()
        print(PEAKS.shape)
        print(MAPS.shape)
        print(METHOD)
        
        if self.SettingsWindow.elbow_rb.isChecked():
            N_STATES = clustering_functions.number_of_clusters(MAPS)
        elif self.SettingsWindow.user_rb.isChecked():
            N_STATES = int(self.SettingsWindow.nmaps.text())*2
        print(N_STATES)
        self.n_maps = int(N_STATES/2)
        
        TOLERANCE = float(self.SettingsWindow.tol.text())
        print(TOLERANCE)
        
        if self.SettingsWindow.randinit_rb.isChecked():
            INITIALIZER = "Random"
        elif self.SettingsWindow.kppinit_rb.isChecked():
            INITIALIZER = "K-Means++"
        INITIAL_CENTERS = clustering_functions.initialize_centers(DATA, MAPS, PEAKS, N_STATES, INITIALIZER)
        print(INITIAL_CENTERS.shape)
        
        METRIC = self.SettingsWindow.arg.currentText()
        print(METRIC)
        
        REPEAT = int(self.SettingsWindow.repeat.text())
        
        clustering_instance, best_maps, gev, final_segmentation = clustering_functions.clustering_func(DATA, MAPS, METHOD,
                                                                                  N_STATES, INITIAL_CENTERS,
                                                                                  REPEAT, TOLERANCE, METRIC)

        self.final_segmentation = final_segmentation
        self._finished_clustering = 1
        
        self.MicrostateDialog.plot_maps(best_maps, gev, clustering_functions.EEG_INFO)
        
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
        
        extracted_features_df = extract_features_functions.extract_features(clustering_functions.FILENAMES, clustering_functions.LENGTH_DATA, Segmentation, Fs, Features)
        print(extracted_features_df)
        save_path = self.foldername.text()
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
        elif METHOD == "X-MEANS":
            self.SettingsWindow.other1.setText("X-Means Splitting Criterion:")
            self.SettingsWindow.arg.clear()
            self.SettingsWindow.arg.addItem("Bayesian Information Criterion")
            self.SettingsWindow.arg.addItem("Minimum Noiseless Description Length")
            self.SettingsWindow.arg.setCurrentText("Bayesian Information Criterion")
        
        
        self.SettingsWindow.setWindowModality(QtCore.Qt.ApplicationModal)
        self.SettingsWindow.show()
        
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Quit",
                                               "Are you sure you want to quit?",
                                               QMessageBox.Yes | 
                                               QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()