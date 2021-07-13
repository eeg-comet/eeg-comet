#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:48:23 2021

Microstate Toolbox GUI

@author: amin
"""

from PyQt5.QtWidgets import QApplication, QStackedWidget
from windows.MainWindow import MainWindow
import sys


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