
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QProgressBar, QApplication, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


# Define a WorkerThread class to run the long-running task
class WorkerThread(QThread):
    progress_updated = pyqtSignal(int, int)

    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def run(self):
        for i in range(self.max_value + 1):
            if self.isInterruptionRequested():  # Check if the thread should be stopped
                break
            self.progress_updated.emit(i, self.max_value)
            self.msleep(100)  # Simulate some work


class ProgressDialog(QDialog):
    """A custom dialog for displaying progress."""

    def __init__(self, parent=None):
        """Initialize the ProgressDialog."""
        super().__init__(parent)
        self.setWindowTitle("Progress")
        self.setWindowModality(Qt.ApplicationModal)

        # Set the initial size of the window
        self.resize(1000, 400)

        # Create widgets
        self.progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_lineedit = QLineEdit()
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_process)  # Connect stop button to stop_process method

        # Set size policy for widgets
        self.progress_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.progress_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.progress_lineedit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.stop_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Set font size for widgets
        font = QFont()
        font.setPointSize(12)
        self.progress_label.setFont(font)
        self.progress_bar.setFont(font)
        self.progress_lineedit.setFont(font)
        self.stop_button.setFont(font)

        # Set up the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_lineedit)
        layout.addWidget(self.stop_button)

        # Variable to track if process is running
        self.running = False
        self.worker_thread = None

    def update_progress(self, value):
        """Update the progress bar with the given value and maximum value."""
        if not self.running:  # Check if process is running
            return
        self.progress_bar.setValue(value)
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

    def start_process(self, max_value):
        """Start the process."""
        if self.running:
            return  # If the process is already running, do nothing
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(max_value)
        self.worker_thread = WorkerThread(max_value)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.start()
        self.running = True

    def stop_process(self):
        """Stop the process."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.requestInterruption()
        self.running = False
        self.close()
