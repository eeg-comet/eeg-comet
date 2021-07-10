#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:48:23 2021

Microstate Toolbox GUI

@author: amin
"""

from PyQt5 import QtWidgets, QtGui, QtCore

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDesktopWidget, QDialog, QProgressBar, QGridLayout
from PyQt5.QtWidgets import QAction, QLabel, QLineEdit, QPushButton, QMessageBox, QStackedWidget, QButtonGroup
from PyQt5.QtWidgets import QGroupBox, QCheckBox, QVBoxLayout, QHBoxLayout, QScrollBar, QRadioButton
from PyQt5.QtWidgets import QWidget, QComboBox, QPushButton, QStyleFactory, QListWidget, QListWidgetItem

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


import sys
import mne
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from itertools import groupby


#from pyclustering import cluster
from pyclustering.cluster import kmeans, xmeans, bsas, clarans, mbsas, optics, rock, elbow
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import os
from fnmatch import fnmatch

### change
import numpy as np
import mne
from scipy.signal import butter, lfilter
import collections
from matplotlib import pyplot as plt
import os
import glob
from scipy.signal import savgol_filter, find_peaks
from sklearn.metrics.pairwise import cosine_similarity

eeglist = []
input_folder="/home/amin/Encfs/TMSEEG_DATA/microstate_toolbox/data/"
pattern="*"
extension=".set"
for path, subdirs, files in os.walk(input_folder):
    for name in files:
        if fnmatch(name, pattern+extension):
            eeglist.append(os.path.join(path, name))
print(eeglist)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def concatenate_files(folder):
    os.chdir(folder)
    extension = 'set'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    
    for file in range(len(all_filenames)):
        print(100*file/len(all_filenames))
        filename = all_filenames[file]
        # Load the example MNE data
        EEG = mne.io.read_raw_eeglab(folder+filename, preload=True, verbose='CRITICAL')
        # Select EEG channels from the dataset
        EEG = EEG.pick_types(meg=False, eeg=True, eog=False, verbose='CRITICAL')
        if file == 0:
            channels = EEG.info['ch_names']
        else:
            channels = np.append(channels,EEG.info['ch_names'])
    
    counter = collections.Counter(channels)
    counter = np.array(list(counter.items()))
    channels2remove = counter[np.where(counter[:,1].astype(float) < len(all_filenames)),0].tolist()

    for file in range(len(all_filenames)):
        print(100*file/len(all_filenames))
        filename = all_filenames[file]
        print('\nLoading EEG Files ... ', filename)
        # Load the example MNE data
        EEG = mne.io.read_raw_eeglab(folder+filename, preload=True, verbose='CRITICAL')
        # Select EEG channels from the dataset    
        EEG = EEG.pick_types(meg=False, eeg=True, eog=False,
                             exclude=channels2remove[0], verbose='CRITICAL')
        EEG = EEG.set_eeg_reference('average')
        
        data_tmp = EEG[:,:][0]
        data_len = data_tmp.shape[1]
        
        #n_channels = EEG.info['nchan']
        eeg_info = EEG.info
        # Sampling Rate
        global Fs
        Fs = 250
        
        if EEG.info['sfreq'] != Fs:
            EEG = EEG.resample(sfreq=Fs)
        if file == 0:
            filenames = filename
            data = data_tmp
            data_length = data_len
        else:
            filenames = np.append(filenames, filename)
            data = np.append(data, data_tmp, axis=1)
            data_length = np.append(data_length, data_len)
    return data, filenames, eeg_info, data_length




'''
# Global Field Potential
gfp = np.std(DATA, axis=0)
# Smooth Data
gfp = utils_microstate.smooth_data(gfp, kernel_size)    
peaks, _ = find_peaks(gfp, distance=Fs*min_dist/1000)
maps = DATA[:, peaks].T
maps /= np.linalg.norm(maps, axis=1, keepdims=True)
'''
### change


###
DATA, FILENAMES, EEG_INFO, LENGTH_DATA = concatenate_files(input_folder)
print(FILENAMES)
n_channels = EEG_INFO['nchan']
# Filter Data
INPUT_DATA = butter_bandpass_filter(DATA,2,20,Fs,order=5)
data_sum_sq = np.sum(DATA ** 2)


## Functions




def _corr_vectors(A, B, axis=0):
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)

def smooth_data(gfp, kernel_size):
    kernel = np.ones(kernel_size)/kernel_size
    smoothed_data = np.convolve(gfp, kernel, mode='same')
    return smoothed_data

def _pre_clustering(data, fs, smoothing):
    # Global Field Potential (GFP)
    gfp = np.std(data, axis=0)
    if smoothing:
        gfp = smooth_data(gfp, smoothing)
    # Find GFP Peaks
    min_dist = 50
    peaks, _ = find_peaks(gfp, distance=fs*min_dist/1000)
    # Create Maps
    maps = data[:, peaks].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    return maps, peaks


def _gev(data, maps):
    gfp = np.std(data, axis=0)
    gfp_sum_sq = np.sum(gfp ** 2)
    if maps.ndim == 1:
        maps /= np.linalg.norm(maps, keepdims=True)
        maps = np.reshape(maps, (1,-1))
    else:
        maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    activation = np.array(maps).dot(data)
    segmentation = np.argmax(np.abs(activation), axis=0)
    map_corr = _corr_vectors(data, maps[segmentation].T)
    gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
    return gev

    
def plot_maps(maps, info):
    maps = np.array(maps)
    plt.figure(figsize=(2 * len(maps), 2))
    for i, map in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        mne.viz.plot_topomap(map, info)
        plt.title('%d' % i)



def concatenate_files(data):
    count = 0
    if count == 0:
        DATA = data
    else:
        DATA = np.append(DATA, data, axis=1)
        count = count+1
    return DATA

def number_of_clusters(maps, cmin=4, cmax=20):
    # create instance of Elbow method using C value from 2 to 10.
    elbow_instance = elbow.elbow(maps, cmin, cmax)
    # process input data and obtain results of analysis
    elbow_instance.process()
    amount_clusters = elbow_instance.get_amount()   # most probable amount of clusters
    #wce = elbow_instance.get_wce()                  # total within-cluster errors for each K
    return amount_clusters

def initialize_centers(data, maps, peaks, n_states, initializer):
    # create instance of K-Means algorithm with prepared centers
    if initializer == 'Random':
        random_state = np.random.RandomState(None)
        chosen_peaks = random_state.choice(len(peaks),size=n_states,replace=False)
        initial_peaks = peaks[chosen_peaks].tolist()
        initial_centers = data[:, initial_peaks].T
    elif initializer == 'K-Means++':
        # Calculate initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(maps,n_states).initialize()
    initial_centers /= np.linalg.norm(initial_centers, axis=1, keepdims=True)
    return initial_centers

def remove_similar_maps(data, centers, clusters):
    sim = abs(cosine_similarity(centers))
    sim = np.triu(sim)
    np.fill_diagonal(sim, 0)
    #similar_maps = np.array(top_n_indexes(sim, N_STATES))
    similar_maps = np.array(np.where(sim>0.90)).transpose()
    final_clusters = np.empty((len(similar_maps),2),dtype=object)
    remove_maps = []
    i = 0
    for s in range(len(similar_maps)):
        to_rm = np.argmin([_gev(data, centers[similar_maps[s,0]]),
                          _gev(data, centers[similar_maps[s,1]])])
        #centers = np.delete(centers, similar_maps[s, to_rm], axis=0)
        
        final_clusters[s,0] = similar_maps[s, 1-to_rm]
        final_clusters[s,1] = np.sort(np.append(clusters[similar_maps[s, 1-to_rm]],
                                         clusters[similar_maps[s, to_rm]]))
        
        remove_maps = np.append(remove_maps, similar_maps[s, to_rm])
        i += 1
        
    remove_maps = np.unique(remove_maps)
    remove_maps = remove_maps.astype(int)
    #print(remove_maps)
    #max_arg = np.unravel_index(sim.argmax(), sim.shape)
    final_centers = np.delete(centers, remove_maps, axis=0)
    return final_centers, final_clusters

def clustering_func(data, maps, method, n_states, initial_centers, repeat, tolerance, metric):
    
    if method == 'K-MEANS':
        if metric == 'Euclidean':
            METRIC = type_metric.EUCLIDEAN
        elif metric == 'Euclidean Square':
            METRIC = type_metric.EUCLIDEAN_SQUARE
        elif metric == 'Manhattan':
            METRIC = type_metric.MANHATTAN
        elif metric == 'Chebyshev':
            METRIC = type_metric.CHEBYSHEV
        elif metric == 'Minkowski':
            METRIC = type_metric.MINKOWSKI
        clustering_instance = kmeans.kmeans(maps, initial_centers,
                                     tolerance=tolerance, itermax=100,
                                     metric=distance_metric(METRIC))
    elif method == 'X-MEANS':
        if metric == 'Bayesian Information Criterion':
            CRITERION = xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION
        elif metric == 'Minimum Noiseless Description Length':
            CRITERION = xmeans.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH 
        clustering_instance = xmeans.xmeans(maps, initial_centers, n_states,
                             tolerance=tolerance, criterion=CRITERION)
    elif method == 'BSAS':
        clustering_instance = bsas.bsas(maps, n_states, tolerance);
    elif method == 'CLARANS':
        clustering_instance = clarans.clarans(maps, n_states, 100, 10);
    elif method == 'MBSAS':
        clustering_instance = mbsas.mbsas(maps, n_states, tolerance);
    elif method == 'OPTICS':
        clustering_instance = optics.optics(maps, 2.0, 3,
                                     amount_of_clusters=n_states);
    elif method == 'ROCK':
        clustering_instance = rock.rock(maps, 1.0, n_states);
    
    
        
    for r in range(repeat):
        print('\nClustering: ', r+1)
        print('Number of Microstate Maps: ', int(n_states/2))
        
        
        centers = np.zeros((n_states, n_channels))
        n = 0
        while centers.shape[0] != int(n_states/2):
            # Run Cluster Analysis
            clustering_instance.process()
            clusters = clustering_instance.get_clusters()
            centers = np.empty((n_states,n_channels))
            for cl in range(len(clusters)):
                centers[cl,:] = np.mean(maps[clusters[cl],:],axis=0)
            # Filter Maps
            centers, clusters = remove_similar_maps(data, centers, clusters)
            
            if n == 5:
                not_converged = True
                print('not converged')
                break
            else:
                not_converged = False
                n += 1
        
        GEV = 0
        if not not_converged:
            GEV_R = _gev(INPUT_DATA, np.array(centers))
            print('GEV = ', GEV_R)
            if GEV_R > GEV:
                GEV = GEV_R
                best_maps = centers
    
    print('\nBest GEV = ', GEV)
    
    activation = np.array(best_maps).dot(DATA)
    final_segmentation = np.argmax(np.abs(activation), axis=0)  
    
    return clustering_instance, best_maps, GEV, final_segmentation

def extract_features(segmentation, fs, features):
    if segmentation is not None:
        extracted_features_df = pd.DataFrame()
        for i in range(len(LENGTH_DATA)):
            if i==0:
                start = 0
                stop = LENGTH_DATA[i]
                stop_pre = stop
            else:
                start = stop_pre
                stop = stop_pre+LENGTH_DATA[i]
                stop_pre = stop
            segment_each = segmentation[start:stop]
            
            extracted_features, headers = [], []
            headers = np.append(headers, "Filename")
            extracted_features = np.append(extracted_features, FILENAMES[i])
            for c in np.unique(segmentation).tolist():
                if "FOC" in features:
                    # Frequency of Occurence for each map per second
                    FOC = 100*np.count_nonzero(segment_each == c)/len(segment_each)
                    headers = np.append(headers, "FOC_"+c)
                    extracted_features = np.append(extracted_features, FOC)
                if "MMD" in features:
                    # Mean Microstates Duration
                    D = [sum(1 for i in g) for k,g in groupby(segment_each) if k==c]
                    MMD = (1000/fs) * (c if not D else sum(D) / len(D))
                    headers = np.append(headers, "MMD_"+c)
                    extracted_features = np.append(extracted_features, MMD)
            extracted_features_df = extracted_features_df.append(pd.DataFrame(extracted_features.reshape(1,len(extracted_features)),
                                                 columns=headers.tolist()))
    return extracted_features_df

def save_features(extracted_features_df, save_path):
    save_name = os.path.join(save_path, 'extracted_features.csv')
    extracted_features_df.to_csv(save_name, index=False, header=True)

class MicrostateFigure(QDialog):
    def __init__(self, parent=None):
        super(MicrostateFigure, self).__init__(parent)
        
        self.setGeometry(500, 500, 700, 500)
        #self.setFixedWidth(700)
        #self.setFixedHeight(340)
        self.setWindowTitle("Microstate Maps")
        
        self.n_maps = None
        self.micro_labels = []

        self.set_layout()
        
        self.manual_labeling_button.clicked.connect(self.manual_micro_label) 
        
    
    '''
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MicrostateFigure, self).__init__(fig)
        
        sc = MicrostateFigure(self, width=3, height=2, dpi=100)
        sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        
        sc.move(150,225)
        sc.setFixedHeight(200)
        sc.setFixedWidth(500)
        
        # Create toolbar
        toolbar = NavigationToolbar(sc, self)
        layout  = QtWidgets.QGridLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        
        self.setCentralWidget(widget)
    '''
    
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
        
        self.labelgev = QLabel(self)
        
        self.label = QLabel(self)
        self.label.setText("Global Explained Variance: ")
        
        
        
    
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
            
        '''
        for i in range(maps.shape[0]):
            self.microlabel = QLineEdit(self)
            self.microlabel.setAlignment(QtCore.Qt.AlignCenter)
            regex = QtCore.QRegExp("[a-z-A-Z]")
            validator = QtGui.QRegExpValidator(regex, self.microlabel)
            self.microlabel.setValidator(validator)
            font = QtGui.QFont("Times", 15, QtGui.QFont.Bold)
            self.microlabel.setFont(font)
            self.microlabel.setMaxLength(1)
            self.microlabel.setFixedWidth(40)
            Layout1.addWidget(self.microlabel)
        '''
        Layout2.addWidget(self.label)
        Layout2.addWidget(self.labelgev)
        
        Layout3.addWidget(self.manual_labeling_button)
        Layout3.addWidget(self.auto_labeling_button)
        
        Layout0.addLayout(Layout1)
        Layout0.addLayout(Layout2)
        Layout0.addLayout(Layout3)
        self.setLayout(Layout0)
        
        self.n_maps = maps.shape[0]
        maps = np.array(maps)
        Figure(figsize=(2 * len(maps), 2))
        for i in range(maps.shape[0]):
            ax = self.figure.add_subplot(1, maps.shape[0], i + 1)
            ax.clear()
            mne.viz.plot_topomap(maps[i,:], info, axes=ax)
        '''
        for i, map in enumerate(maps):
            ax = self.figure.add_subplot(1, len(maps), i + 1)
            ax.clear()
            #ax.subplot(1, len(maps), i + 1)
            mne.viz.plot_topomap(map, info)
            #ax.title('%d' % i)
        '''
        self.labelgev.setText(str(gev))
        self.labelgev.setFixedWidth(300)
        
        self.canvas.draw()
    
        
    def manual_micro_label(self):
        for i in range(self.n_maps):
            microlabel_attr = getattr(self, "microlabel{}".format(i))
            self.micro_labels.append(microlabel_attr.text())
        print(self.micro_labels)
        ClusteringWindow.micro_labels = self.micro_labels
        self.close()
        
    def displayFigure(self):
        self.show()


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
        

class ClusteringWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ClusteringWindow, self).__init__(parent)
        
        self.setGeometry(500, 500, 1000, 1000)
        self.setWindowTitle("Microstate Feature Extraction")
        
        
        self.ClusteringWindowUI()
        
        self.SettingsWindow = SettingsWindow()
        self.MicrostateDialog = MicrostateFigure()
        
        self._finished_clustering = 0
        
        #self.labeling_button.clicked.connect(self.filter_data)
        
        self.browse_button.clicked.connect(self.browsefiles)
        
        self.settings_button.clicked.connect(self.passingInformation)
        
        #self.micro_labels = None
        self.n_maps = None
        self.final_segmentation = None
        
        self.process_button.clicked.connect(self.do_clustering)
        self.process_button.clicked.connect(self.plot_micro)
        
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
        
        
        #self.back.clicked.connect(self.plot_micro)
        
        
        
        
    # Functions
    
    def plot_micro(self):
        self.MicrostateDialog.displayFigure()
        
        
        
        
    def browsefiles(self):
        fname = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.foldername.setText(fname)
        
            
    def do_clustering(self):
        
        Fs = 250 #from previous window
        DATA = INPUT_DATA #from previous window
        print(self.SettingsWindow.smooth_controller())
        if self.SettingsWindow.smooth_controller():
            SMOOTHING = int(self.SettingsWindow.kernel_size.text())
        else:
            SMOOTHING = []
        print(SMOOTHING)
        MAPS, PEAKS = _pre_clustering(DATA, Fs, SMOOTHING)
        METHOD = self.clustering_method.currentText()
        print(PEAKS.shape)
        print(MAPS.shape)
        print(METHOD)
        
        if self.SettingsWindow.elbow_rb.isChecked():
            N_STATES = number_of_clusters(MAPS)
        elif self.SettingsWindow.user_rb.isChecked():
            N_STATES = int(self.SettingsWindow.nmaps.text())*2
        print(N_STATES)
        self.n_maps = int(N_STATES/2)
        
        TOLERANCE = float(self.SettingsWindow.tol.text())
        print(TOLERANCE)
        
        if self.SettingsWindow.randinit_rb.isChecked():
            INITIALIZER = "Random"
            INITIAL_CENTERS = initialize_centers(DATA, MAPS, PEAKS, N_STATES, INITIALIZER)
        elif self.SettingsWindow.kppinit_rb.isChecked():
            INITIALIZER = "K-Means++"
            INITIAL_CENTERS = initialize_centers(DATA, MAPS, PEAKS, N_STATES, INITIALIZER)
        print(INITIAL_CENTERS.shape)
        
        METRIC = self.SettingsWindow.arg.currentText()
        print(METRIC)
        
        REPEAT = int(self.SettingsWindow.repeat.text())
        
        clustering_instance, best_maps, gev, final_segmentation = clustering_func(DATA, MAPS, METHOD,
                                                                                  N_STATES, INITIAL_CENTERS,
                                                                                  REPEAT, TOLERANCE, METRIC)

        self.final_segmentation = final_segmentation
        self._finished_clustering = 1
        
        self.MicrostateDialog.plot_maps(best_maps, gev, EEG_INFO)
        
    
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
        
        extracted_features_df = extract_features(Segmentation, Fs, Features)
        print(extracted_features_df)
        save_path = self.foldername.text()
        # Save Features
        save_features(extracted_features_df, save_path)
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
        
        
        self.SettingsWindow.displaySettings()
    
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
    mainwindow = ClusteringWindow()
    widget = QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(800)
    widget.setFixedHeight(800)
    
    widget.show()
    sys.exit(app.exec_())
    
window()