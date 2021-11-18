#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:35:44 2021

@author: amin
"""
import os.path

from PyQt5 import QtGui, QtCore

from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QLineEdit
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import os.path
import numpy as np
import mne
from matplotlib import pyplot as plt
from functions.test_classifier import label_micromap

#from windows.ClusteringWindow import ClusteringWindow

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
        
        # Labeling Button
        self.manual_labeling_button = QPushButton(self)
        self.manual_labeling_button.setText("manual labeling")
        self.manual_labeling_button.setFixedHeight(50)
        
        self.auto_labeling_button = QPushButton(self)
        self.auto_labeling_button.setText("automatic labeling")
        self.auto_labeling_button.setFixedHeight(50)
        
        self.finish_button = QPushButton(self)
        self.finish_button.setText("save labels")
        self.finish_button.setFixedHeight(50)
        
        self.labelgev = QLabel(self)
        
        self.label = QLabel(self)
        self.label.setText("Global Explained Variance: ")
        
        
        
    def plot_maps(self, maps, gev, info):
        
        # Set the Layout
        self.Layout0 = QVBoxLayout()
        self.Layout1 = QHBoxLayout()
        self.Layout2 = QHBoxLayout()
        self.Layout3 = QVBoxLayout()
        
        self.Layout0.addWidget(self.toolbar)
        self.Layout0.addWidget(self.canvas)
        
        for i in range(maps.shape[0]):
            exec(f'self.microlabel{i} = QLineEdit(self)')
            microlabel_attr = getattr(self, "microlabel{}".format(i))
            self.Layout1.addWidget(microlabel_attr)
            microlabel_attr.setAlignment(QtCore.Qt.AlignCenter)
            regex = QtCore.QRegExp("[a-z-A-Z]")
            validator = QtGui.QRegExpValidator(regex, microlabel_attr)
            microlabel_attr.setValidator(validator)
            font = QtGui.QFont("Times", 15, QtGui.QFont.Bold)
            microlabel_attr.setFont(font)
            microlabel_attr.setMaxLength(1)
            microlabel_attr.setFixedWidth(40)
            
        self.Layout2.addWidget(self.label)
        self.Layout2.addWidget(self.labelgev)
        
        self.Layout3.addWidget(self.manual_labeling_button)
        self.Layout3.addWidget(self.auto_labeling_button)
        self.Layout3.addWidget(self.finish_button)
        
        self.Layout0.addLayout(self.Layout1)
        self.Layout0.addLayout(self.Layout2)
        self.Layout0.addLayout(self.Layout3)
        self.setLayout(self.Layout0)
        
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

        self.labelgev.setText(str(gev))
        self.labelgev.setFixedWidth(300)
        
        self.canvas.draw()
    
    def manual_micro_label(self):
        for i in range(self.n_maps):
            microlabel_attr = getattr(self, "microlabel{}".format(i))
            self.micro_labels.append(microlabel_attr.text())
        self.manual_labeling_button.setStyleSheet("background-color: green")
        self.micro_labeled = True
        #ClusteringWindow.micro_labels = self.micro_labels
        #self.close()
    
    def auto_micro_label(self):
        self.micro_labels = []
        for i in range(self.n_maps):
            y_pred = label_micromap(str(i)+'.png')
            self.micro_labels.append(y_pred)
            
            microlabel_attr = getattr(self, "microlabel{}".format(i))
            microlabel_attr.setText(y_pred)
        self.micro_labeled = True
    
    def pass_labels(self):
        if self.micro_labeled:
            self.micro_labels
            self.close()