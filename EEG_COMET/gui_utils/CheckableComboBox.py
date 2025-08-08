"""Custom QComboBox with checkbox items for multi-select."""

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QFontMetrics, QPalette, QStandardItem
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate, qApp


class CheckableComboBox(QComboBox):
    """Custom QComboBox that allows selecting multiple items with checkboxes."""

    # Define a custom signal
    itemCheckedStateChanged = pyqtSignal(int, bool)

    class Delegate(QStyledItemDelegate):
        """Delegate to increase row height for better readability."""
        def sizeHint(self, option, index):
            """Return a taller size hint for each row."""
            size = super().sizeHint(option, index)
            size.setHeight(40)
            return size

    def __init__(self, *args, **kwargs):
        """Initialize the CheckableComboBox instance."""
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
        """Update displayed text and elide as needed on resize."""
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):
        """Handle events on the line edit and the view's viewport."""
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

            # Store previous state
            item.checkState()

            if item.checkState() == Qt.Checked:
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)

            # Emit signal with index and new checked state
            is_checked = item.checkState() == Qt.Checked
            self.itemCheckedStateChanged.emit(index.row(), is_checked)

            return True
        return False

    def showPopup(self):
        """Show the popup and enable clicking the line edit to close it."""
        super().showPopup()
        self.closeOnLineEditClick = True

    def hidePopup(self):
        """Hide the popup and temporarily disable immediate reopening."""
        super().hidePopup()
        self.startTimer(100)
        self.updateText()

    def timerEvent(self, event):
        """Delay reopening the popup via timer event handler."""
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        """Update line edit text based on selected items."""
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
        """Add an item with text and optional data."""
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
        """Add multiple items with the given texts and optional data."""
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def deselectAllItems(self):
        """Deselect all items in the combo box."""
        model = self.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            item.setCheckState(Qt.Unchecked)

    def currentData(self):
        """Return data of currently selected items."""
        return [
            self.model().item(i).data()
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.Checked
        ]
