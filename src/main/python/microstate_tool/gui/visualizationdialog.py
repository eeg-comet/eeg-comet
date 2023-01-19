
import os.path
import pandas as pd
import seaborn as sns

from PyQt5 import uic
#from PyQt5 import QtWebEngineWidgets
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout

from functions.utils.load_save_config import load_config
import plotly.express as px

class VisualizationDialog(QDialog):
    def __init__(self, context, parent=None):
        super(VisualizationDialog, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        self.ui = uic.loadUi(context.get_resource("VisualizationWindow.ui"), self)

        self.ui.setWindowTitle("Visualization of the extracted features")

        self.ui.plot_all_button.clicked.connect(self.show_boxplots_all)
        self.ui.plot_groups_button.clicked.connect(self.show_boxplots_groups)

        self.ui.add_group_a_button.clicked.connect(self.files2a)
        self.ui.add_group_b_button.clicked.connect(self.files2b)
        self.ui.remove_group_a_button.clicked.connect(self.a2files)
        self.ui.remove_group_b_button.clicked.connect(self.b2files)
        self.ui.reset_groups_button.clicked.connect(self.reset_groups)
        self.resize(1000, 800)

    def files2a(self):
        if self.ui.all_files_list.currentItem():
            filename = self.ui.all_files_list.currentItem().text()
            listItems = self.all_files_list.selectedItems()
            if not listItems: return
            for item in listItems:
                self.all_files_list.takeItem(self.all_files_list.row(item))

            self.ui.group_a_files_list.addItem(str(filename))

    def files2b(self):
        if self.ui.all_files_list.currentItem():
            filename = self.ui.all_files_list.currentItem().text()
            listItems = self.all_files_list.selectedItems()
            if not listItems: return
            for item in listItems:
                self.all_files_list.takeItem(self.all_files_list.row(item))

            self.ui.group_b_files_list.addItem(str(filename))

    def a2files(self):
        if self.ui.group_a_files_list.currentItem():
            filename = self.ui.group_a_files_list.currentItem().text()
            listItems = self.group_a_files_list.selectedItems()
            if not listItems: return
            for item in listItems:
                self.group_a_files_list.takeItem(self.group_a_files_list.row(item))

            self.ui.all_files_list.addItem(str(filename))

    def b2files(self):
        if self.ui.group_b_files_list.currentItem():
            filename = self.ui.group_b_files_list.currentItem().text()
            listItems = self.group_b_files_list.selectedItems()
            if not listItems: return
            for item in listItems:
                self.group_b_files_list.takeItem(self.group_b_files_list.row(item))

            self.ui.all_files_list.addItem(str(filename))

    def reset_groups(self):
        self.ui.group_a_files_list.clear()
        self.ui.group_b_files_list.clear()
        self.load_filenames()

    def load_filenames(self):
        # Load config
        config_file = os.path.join(self.save_folder, 'log.ini')
        config = load_config(config_file)
        # Load "feature extraction settings" from config
        features2extract_str = config['feature extraction settings']['features2extract']
        self.features2extract = features2extract_str.split(",")

        # Load extracted features
        self.extracted_features_df = pd.read_csv(os.path.join(self.extracted_features_path,
                                                         'extracted_features.csv'))

        filenames = self.extracted_features_df['Filename']
        for i in range(len(filenames)):
            self.ui.all_files_list.addItem(str(filenames[i]))

    def show_boxplots_all(self):
        feature = self.ui.feature_combo.currentText()
        if feature == "Coverage":
            self.feature = 'COV'
        elif feature == "Frequency of Occurrence":
            self.feature = 'FOC'
        elif feature == "Mean Microstate Duration":
            self.feature = 'MMD'
        elif feature == "Global Explained Variance":
            self.feature = 'GEV'
        elif feature == "Microstate Complexity":
            self.feature = 'LZC'
        elif feature == "Transition Probability":
            self.feature = 'TP'

        filter_col = [col for col in self.extracted_features_df if col.startswith(self.feature)]
        filter_col.sort()

        df_features = pd.melt(self.extracted_features_df.reset_index(),
                              id_vars=['Filename'],
                              value_vars=filter_col)
        df_features.columns = ['Filename', 'Feature', self.feature]

        ax = self.ui.MplWidget_boxplots.canvas.axes
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
        boxplot = sns.boxplot(x='Feature', y=self.feature, data=df_features, ax=ax)
        self.ui.MplWidget_boxplots.canvas.draw()

    def show_boxplots_groups(self):
        feature = self.ui.feature_combo.currentText()
        if feature == "Coverage":
            self.feature = 'COV'
        elif feature == "Frequency of Occurrence":
            self.feature = 'FOC'
        elif feature == "Mean Microstate Duration":
            self.feature = 'MMD'
        elif feature == "Global Explained Variance":
            self.feature = 'GEV'
        elif feature == "Microstate Complexity":
            self.feature = 'LZC'
        elif feature == "Transition Probability":
            self.feature = 'TP'

        group_a_name = self.ui.group_a_lineedit.text()
        group_b_name = self.ui.group_b_lineedit.text()

        filter_col = [col for col in self.extracted_features_df if col.startswith(self.feature)]
        filter_col.sort()

        df_features = pd.melt(self.extracted_features_df.reset_index(),
                              id_vars=['Filename'],
                              value_vars=filter_col)
        df_features.columns = ['Filename', 'Feature', self.feature]

        self.listItems_group_a = []
        for x in range(self.ui.group_a_files_list.count()):
            self.listItems_group_a.append(self.ui.group_a_files_list.item(x).text())
        self.listItems_group_b = []
        for x in range(self.group_b_files_list.count()):
            self.listItems_group_b.append(self.ui.group_b_files_list.item(x).text())

        df_features_a = df_features[df_features['Filename'].isin(self.listItems_group_a)]
        df_features_a['Group'] = [group_a_name] * len(df_features_a)
        df_features_b = df_features[df_features['Filename'].isin(self.listItems_group_b)]
        df_features_b['Group'] = [group_b_name] * len(df_features_b)
        df_features = pd.concat([df_features_a, df_features_b], axis=0).reset_index()

        ax = self.ui.MplWidget_boxplots.canvas.axes
        ax.clear()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
        boxplot = sns.boxplot(x='Feature', y=self.feature, hue='Group', data=df_features, ax=ax)
        self.ui.MplWidget_boxplots.canvas.draw()
