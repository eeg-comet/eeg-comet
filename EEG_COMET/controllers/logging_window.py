
import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from datetime import datetime


class Worker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value
        self.stopped = False  # Flag to track thread stop request

    def run(self):
        for i in range(self.max_value):
            if self.stopped:  # Check if the thread should be stopped
                break  # Exit the loop if stop is requested
            self.progress_updated.emit(i)
            self.msleep(100)
        if not self.stopped:
            self.finished.emit()

    def stop(self):
        """Stop the thread."""
        self.stopped = True


class LogWindow(QWidget):
    stop_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        script_path = os.path.abspath(__file__)
        ui_path = os.path.join(os.path.dirname(script_path), "..", "ui", "LogWindow.ui")
        self.ui = uic.loadUi(ui_path, self)
        self.ui.setWindowTitle("EEG-COMET Log")
        self.ui.progress_stop_button.clicked.connect(self.stop_process)
        self.running = False
        self.worker_thread = None

    def append_log(self, log, log_type='info'):
        """Appends a log entry with the current date and time to the log_text list."""
        current_date = datetime.now().strftime("%d/%m/%y")
        current_time = datetime.now().strftime("%I:%M %p")
        if log_type == 'settings':
            separator = "*" * 50
            current_log_text = f"\n{separator}\n{log}\n{separator}\n"
        else:
            current_log_text = f"[{current_date} {current_time}]: {log}\n"
        self.ui.log_text_area.append(current_log_text)

    def replace_log(self, log_text):
        """Replace the current log with the imported log."""
        self.ui.log_text_area.setText(log_text)

    def setup_progress_dialog(self, window_title, label_text, max_value):
        """Sets up a progress dialog with the specified window title, label text, and maximum value."""
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.show()
        self.setWindowTitle(window_title)
        self.ui.progress_label.setText(label_text)
        self.ui.progress_bar.setValue(0)
        self.ui.progress_bar.setRange(0, max_value)
        self.running = True
        self.worker_thread = Worker(max_value)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.stop_requested.connect(self.worker_thread.stop)
        self.worker_thread.start()

    def update_progress(self, value, text=None):
        """Update the progress bar with the given value and maximum value."""
        if not self.running:  # Check if the process is running
            return
        if text is not None:
            self.set_line_edit_text(text)
        self.ui.progress_bar.setValue(value + 1)
        QApplication.processEvents()

    def process_finished(self, text=None):
        """Called when the processing is finished."""
        self.setWindowFlags(Qt.Window)
        self.show()
        if text is not None:
            self.set_line_edit_text(text)
        self.running = False
        self.ui.progress_stop_button.setText("Stop Processing")

    def set_window_title(self, title):
        """Set the window title."""
        self.setWindowTitle(title)

    def set_label_text(self, text):
        """Set the text of the label."""
        self.ui.progress_label.setText(text)

    def set_line_edit_text(self, text):
        """Set the text of the line edit."""
        self.ui.progress_lineedit.setText(text)

    def stop_process(self):
        """Stop the process."""
        if self.worker_thread and self.worker_thread.isRunning() and self.running:
            self.stop_requested.emit()
            self.worker_thread.quit()
        self.running = False

    def show_hide_log_window(self):
        """Toggle the visibility of the log window."""
        self.setVisible(not self.isVisible())
