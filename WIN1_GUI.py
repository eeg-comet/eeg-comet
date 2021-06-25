#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:48:23 2021

Microstate Toolbox GUI

@author: amin
"""

from PyQt5 import QtGui, QtCore

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QMessageBox, QComboBox
from PyQt5.QtWidgets import QScrollBar, QStackedWidget, QRadioButton, QProgressBar


import sys
import os
import numpy as np
import mne
import collections
import h5py
from fnmatch import fnmatch
from scipy.signal import butter, lfilter, resample


# functions
def find_eeg_func(input_folder, extension, pattern):
    list_eegs = []
    for path, subdirs, files in os.walk(input_folder):
        for name in files:
            if fnmatch(name, pattern+extension):
                list_eegs.append(os.path.join(path, name))
    return list_eegs

def filter_eeg_func(data, fs, lowcut, highcut, order):
    nyq = 0.5 * fs
    l_freq = lowcut / nyq
    h_freq = highcut / nyq
    b, a = butter(order, [l_freq, h_freq], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data

def chan2rm_eeg_func(list_eegs):
    # Find EEG channels that all files have
    for file in range(len(list_eegs)):
        print(100*file/len(list_eegs))
        filename = list_eegs[file]
        # Load the example MNE data
        EEG = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
        # Select EEG channels from the dataset
        EEG = EEG.pick_types(meg=False, eeg=True, eog=False, verbose='CRITICAL')
        if file == 0:
            channels = EEG.info['ch_names']
        else:
            channels = np.append(channels,EEG.info['ch_names'])
    counter = collections.Counter(channels)
    counter = np.array(list(counter.items()))
    channels2remove = counter[np.where(counter[:,1].astype(float) < len(list_eegs)),0].tolist()
    return channels2remove

def preprocess_eeg_func(filename, channels2remove, filter_bool, lowcut, highcut, order,
                        downsample_bool, fs):
    #for file in range(len(list_eegs)):
    #print(100*file/len(list_eegs))
    #filename = list_eegs[file]
    print('\nLoading EEG Files ... ', filename)
    # Load the example MNE data
    EEG = mne.io.read_raw_eeglab(filename, preload=True, verbose='CRITICAL')
    # Select EEG channels from the dataset    
    EEG = EEG.pick_types(meg=False, eeg=True, eog=False,
                         exclude=channels2remove[0], verbose='CRITICAL')
    EEG = EEG.set_eeg_reference('average')
    
    #n_channels = EEG.info['nchan']
    #INFO = EEG.info
    
    DATA = EEG[:,:][0]        
    if filter_bool:
        DATA = filter_eeg_func(DATA, fs, lowcut, highcut, order)
        
    if downsample_bool:
        if EEG.info['sfreq'] != fs:
            n_samples = round(len(DATA)*float(fs)/EEG.info['sfreq'])
            DATA = resample(DATA, n_samples)
    
    return DATA

def save_preprocessed_eeg_func(data, filename, save_folder):
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    with h5py.File(save_folder+'/'+name+'.h5','w') as f:
        f.create_dataset(name, data=data)
        

# GUI
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
        
        channels2remove = chan2rm_eeg_func(list_eegs)
        
        for filename in list_eegs:
            DATA = preprocess_eeg_func(filename, channels2remove,
                                self.__filter_true, self.lowcut.text(),
                                self.highcut.text(), self.order.text(),
                                self.__downsample_true, self.sample_rate.text())
            
            save_preprocessed_eeg_func(DATA, filename, self.foldername.text())
            self.progress.setValue(100*(list_eegs.index(filename)+1)/len(list_eegs))
        self.__finished = 1
        
    def signal_end(self):
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
        
        
        '''
        # Menu Bar
        quit = QAction("Quit", self)
        quit.triggered.connect(self.close)
        menubar = self.menuBar()
        fmenu = menubar.addMenu("File")
        fmenu.addAction(quit)
        '''
        
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
        list_eegs = find_eeg_func(input_folder, extension, pattern)

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
    mainwindow = MainWindow()
    widget = QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(800)
    widget.setFixedHeight(800)
    
    widget.show()
    sys.exit(app.exec_())
    
window()