#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:35:44 2021

@author: amin
"""
import os.path

from PyQt5 import QtGui, QtCore

from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QLineEdit, QLCDNumber
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy
from configparser import ConfigParser
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import os.path
import numpy as np
import mne
from matplotlib import pyplot as plt

from functions.test_classifier import label_micromap
from functions.utils.load_save_config import load_config, save_config

class MicrostateDialog(QDialog):
    def __init__(self, parent=None):
        super(MicrostateDialog, self).__init__(parent)
        
        self.setGeometry(500, 500, 700, 500)
        #self.setFixedWidth(700)
        #self.setFixedHeight(340)
        self.setWindowTitle("Microstate Maps")

        self.save_dir = ""
        self.micro_labeled = False
        self.n_maps = None
        self.micro_labels = []

        self.set_layout()
        
        self.manual_labeling_button.clicked.connect(self.manual_micro_label)
        self.auto_labeling_button.clicked.connect(self.auto_micro_label)
        
        self.finish_button.clicked.connect(self.pass_labels)
    
    def set_layout(self):
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.lcd_gev = QLCDNumber(self)

        # Labeling Button
        self.manual_labeling_button = QPushButton(self)
        self.manual_labeling_button.setText("manual labeling")
        self.manual_labeling_button.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding)

        self.auto_labeling_button = QPushButton(self)
        self.auto_labeling_button.setText("automatic labeling")
        self.auto_labeling_button.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding)

        self.finish_button = QPushButton(self)
        self.finish_button.setText("save labels")
        self.finish_button.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding)

        self.labelgev = QLabel(self)
        self.labelgev.setText("Global Explained Variance: ")
        
        
        
    def plot_maps(self, maps, gev, info):
        
        # Set the Layout
        Layout0 = QVBoxLayout()
        Layout1 = QHBoxLayout()
        Layout2 = QHBoxLayout()
        Layout3 = QVBoxLayout()
        
        Layout0.addWidget(self.toolbar)
        Layout0.addWidget(self.canvas)
        
        for i in range(maps.shape[0]):
            exec(f'self.microlabel{i} = QLineEdit(self)')
            microlabel_attr = getattr(self, "microlabel{}".format(i))
            Layout1.addWidget(microlabel_attr)
            microlabel_attr.setAlignment(QtCore.Qt.AlignCenter)
            regex = QtCore.QRegExp("[a-z-A-Z]")
            validator = QtGui.QRegExpValidator(regex, microlabel_attr)
            microlabel_attr.setValidator(validator)
            font = QtGui.QFont("Times", 15, QtGui.QFont.Bold)
            microlabel_attr.setFont(font)
            microlabel_attr.setMaxLength(1)
            microlabel_attr.setFixedWidth(40)
            
        Layout2.addWidget(self.labelgev)
        Layout2.addWidget(self.lcd_gev)

        Layout3.addWidget(self.manual_labeling_button)
        Layout3.addWidget(self.auto_labeling_button)
        Layout3.addWidget(self.finish_button)
        
        Layout0.addLayout(Layout1)
        Layout0.addLayout(Layout2)
        Layout0.addLayout(Layout3)
        self.setLayout(Layout0)
        
        self.n_maps = maps.shape[0]
        maps = np.array(maps)
        #Figure(figsize=(2 * len(maps), 2))

        plt.ion()
        for i in range(maps.shape[0]):
            fig = plt.figure()
            mne.viz.plot_topomap(maps[i,:], info, sensors=False)
            fig.savefig(os.path.join(self.save_dir, str(i)+'.png'), bbox_inches='tight', dpi=80)
            fig.show()
            plt.close(fig)
        
        for i in range(maps.shape[0]):
            ax = self.figure.add_subplot(1, maps.shape[0], i + 1)
            ax.clear()
            mne.viz.plot_topomap(maps[i,:], info, sensors=False, axes=ax)
            #self.figure.savefig(str(i)+'.png', bbox_inches='tight', dpi=200)

        self.lcd_gev.display(100*gev)
        self.labelgev.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Expanding)
        # save config
        config = ConfigParser()
        config_file = os.path.join(self.save_dir, 'log.ini')
        config.read(config_file)
        if config.has_section('clustering results'):
            config.remove_section('clustering results')
        config.add_section('clustering results')
        config['clustering results']['gev'] = str(gev)
        save_config(config_file, config)

        self.canvas.draw()
    
    def manual_micro_label(self):
        for i in range(self.n_maps):
            microlabel_attr = getattr(self, "microlabel{}".format(i))
            self.micro_labels.append(microlabel_attr.text())
        self.manual_labeling_button.setStyleSheet("background-color: green")
        # Write microstates labels to config
        config_file = os.path.join(self.save_dir, 'log.ini')
        config = load_config(config_file)
        str_micro_labels = ','.join(map(str, self.micro_labels))
        config['clustering results']['micro_labels'] = str_micro_labels
        config['progress']['done_labeling_microstates'] = str(True)
        save_config(config_file, config)
        #ClusteringWindow.micro_labels = self.micro_labels
        #self.close()
    
    def auto_micro_label(self):
        self.micro_labels = []
        for i in range(self.n_maps):
            y_pred = label_micromap(str(i)+'.png')
            self.micro_labels.append(y_pred)
            
            microlabel_attr = getattr(self, "microlabel{}".format(i))
            microlabel_attr.setText(y_pred)

            # Write microstates labels to config
            config_file = os.path.join(self.save_dir, 'log.ini')
            config = load_config(config_file)
            str_micro_labels = ','.join(map(str, self.micro_labels))
            config['clustering results']['micro_labels'] = str_micro_labels
            config['progress']['done_labeling_microstates'] = str(True)
            save_config(config_file, config)

    def pass_labels(self):
        if self.micro_labeled:
            self.micro_labels
            self.close()