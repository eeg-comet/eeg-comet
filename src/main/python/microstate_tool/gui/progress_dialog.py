
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QProgressBar, QApplication, QSizePolicy
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class ProgressDialog(QDialog):
    """A custom dialog for displaying progress."""

    def __init__(self, parent=None):
        """Initialize the ProgressDialog."""
        super().__init__(parent)
        self.setWindowTitle("Progress")
        self.setWindowModality(Qt.ApplicationModal)

        # Set the initial size of the window
        self.resize(600, 200)

        # Create widgets
        self.progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_lineedit = QLineEdit()
        self.progress_lineedit.setReadOnly(True)

        # Set size policy for widgets
        self.progress_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.progress_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.progress_lineedit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Set font size for widgets
        font = QFont()
        font.setPointSize(12)
        self.progress_label.setFont(font)
        self.progress_bar.setFont(font)
        self.progress_lineedit.setFont(font)

        # Set initial values for progress bar and line edit
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(1)
        self.progress_lineedit.setText("0")

        # Set up the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_lineedit)

    def update_progress(self, value, max_value):
        """Update the progress bar with the given value and maximum value."""
        self.progress_bar.setValue(value)
        self.progress_bar.setMaximum(max(1, max_value))  # Avoid division by zero
        QApplication.processEvents()

    def set_window_title(self, title):
        """Set the window title."""
        self.setWindowTitle(title)

    def set_label_text(self, text):
        """Set the text of the label."""
        self.progress_label.setText(text)

    def set_line_edit_text(self, text):
        """Set the text of the line edit."""
        self.progress_lineedit.setText(text)
