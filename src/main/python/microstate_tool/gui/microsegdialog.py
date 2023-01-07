#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:35:44 2021

@author: amin
"""

from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QComboBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from functions.utils.find_data import find_data
import matplotlib.pyplot as plt
import os.path
import h5py
from functions.utils.visualize_micro_segments import estimate_fc_mat


class MicroSegDialog(QDialog):
    def __init__(self, parent=None):
        super(MicroSegDialog, self).__init__(parent)

        self.setGeometry(500, 500, 700, 500)
        self.setWindowTitle("Visualizing Microstate Segments")

        self.save_dir = ""
        self.set_layout()

        self.start_button.clicked.connect(self.plot_fc_seg)

    def set_layout(self):
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Set the Layout
        Layout0 = QVBoxLayout()
        Layout1 = QHBoxLayout()

        Layout0.addWidget(self.toolbar)
        Layout0.addWidget(self.canvas)

        self.method_label = QLabel(self)
        self.method_label.setText("Functional Connectivity Method: ")
        self.method_label.setFont(QFont('Times', 14))
        Layout1.addWidget(self.method_label)

        self.method_combobox = QComboBox(self)
        self.method_combobox.addItem("Coherence")
        self.method_combobox.addItem("Correlation")
        self.method_combobox.addItem("Cosine")
        self.method_combobox.addItem("Imaginary Coherence")
        self.method_combobox.addItem("Phase Locking Value")
        self.method_combobox.addItem("Imaginary part of Phase Locking Value")
        self.method_combobox.addItem("Phase Lag Index")
        self.method_combobox.setCurrentText("Phase Locking Value")
        self.method_combobox.setFont(QFont('Times', 14))
        Layout1.addWidget(self.method_combobox)

        Layout0.addLayout(Layout1)

        # Labeling Button
        self.start_button = QPushButton(self)
        self.start_button.setText("start")
        self.start_button.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Maximum)
        self.start_button.setFont(QFont('Times', 14))
        Layout0.addWidget(self.start_button)

        self.setLayout(Layout0)


    def plot_fc_seg(self):
        print("Estimating connectivity matrix ...")

        get_method = self.method_combobox.currentText()
        if get_method == "Coherence":
            fc_method = "COH"
        elif get_method == "Correlation":
            fc_method = "CORR"
        elif get_method == "Cosine":
            fc_method = "COS"
        elif get_method == "Imaginary Coherence":
            fc_method = "ICOH"
        elif get_method == "Phase Locking Value":
            fc_method = "PLV"
        elif get_method == "Imaginary part of Phase Locking Value":
            fc_method = "IPLV"
        elif get_method == "Phase Lag Index":
            fc_method = "PLI"

        micro_segments_list = find_data(self.micro_segments_path, '.hdf', '*')

        for s in range(len(micro_segments_list)):
            seg_dir = micro_segments_list[s]
            hf = h5py.File(seg_dir, 'r')
            name = os.path.basename(seg_dir)
            name = os.path.splitext(name)[0]
            eeg_data = hf[name]
            ax = self.figure.add_subplot(1, self.number_of_maps, s + 1)
            ax.clear()

            portion = 100000
            connectivity_matrix = estimate_fc_mat(eeg_data[:,:portion], fc_method, self.ch_names)

            ax.imshow(connectivity_matrix, interpolation='none')
            plt.show()

        self.canvas.draw()

