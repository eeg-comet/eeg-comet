
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate, qApp
from PyQt5.QtGui import QPalette, QFontMetrics, QStandardItem
from PyQt5.QtCore import Qt, QEvent


class CheckableComboBox(QComboBox):
    """
    The CheckableComboBox class is a custom QComboBox that allows for selecting multiple items with checkboxes.
    """

    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(40)
            return size

    def __init__(self, *args, **kwargs):
        """
        Initialize the CheckableComboBox instance.
        """

        super().__init__(*args, **kwargs)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)
        self.setItemDelegate(CheckableComboBox.Delegate())
        self.model().dataChanged.connect(self.updateText)
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False
        self.view().viewport().installEventFilter(self)
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

    def resizeEvent(self, event):
        """
        Reimplement the resizeEvent method to update the displayed text and elide as needed.
        """

        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):
        """
        Reimplement the eventFilter method to handle events on the line edit and the view's viewport.
        """

        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False
        if object == self.view().viewport() and event.type() == QEvent.MouseButtonRelease:
            index = self.view().indexAt(event.pos())
            item = self.model().item(index.row())
        
            if item.checkState() == Qt.Checked:
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
            return True
        return False

    def showPopup(self):
        """
        Reimplement the showPopup method to show the popup and enable click on the line edit to close it.
        """

        super().showPopup()
        self.closeOnLineEditClick = True

    def hidePopup(self):
        """
        Reimplement the hidePopup method to hide the popup and disable immediate reopening.
        """

        super().hidePopup()
        self.startTimer(100)
        self.updateText()

    def timerEvent(self, event):
        """
        Reimplement the timerEvent method to handle the timer event for delaying reopening the popup.
        """

        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        """
        Update the displayed text in the line edit based on the selected items.
        """

        texts = [
            self.model().item(i).text()
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.Checked
        ]
        text = ", ".join(texts)
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        """
        Add an item to the combo box with the given text and optional data.
        """

        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        """
        Add multiple items to the combo box with the given texts and optional data.
        """

        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def deselectAllItems(self):
        """
        Deselect all items in the combo box.
        """

        model = self.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            item.setCheckState(Qt.Unchecked)

    def currentData(self):
        """
        Get the data of the currently selected items in the combo box.
        """

        return [
            self.model().item(i).data()
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.Checked
        ]
