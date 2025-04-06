
import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from datetime import datetime


class Worker(QThread):
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(str)

    def __init__(self, tasks, processing_func):
        """
        param tasks: A list of tasks. Each task can be any structure (e.g., a tuple, list, or single value)
                      that processing_func can handle.
        param processing_func: A callable that processes a task. It should accept the task (or unpacked values
                                if the task is a tuple or list).
        """
        super().__init__()
        if isinstance(tasks, int):
            self.tasks = range(tasks)
        else:
            self.tasks = tasks
        self.processing_func = processing_func
        self.stopped = False

    def run(self):
        total_tasks = len(self.tasks)
        for idx, task in enumerate(self.tasks, start=1):
            if self.stopped:
                self.finished.emit("Process stopped by user!")
                return

            # If the task is a tuple or list (but not a string), unpack it.
            if isinstance(task, (tuple, list)) and not isinstance(task, str):
                self.processing_func(*task)
            else:
                self.processing_func(task)

            self.progress_updated.emit(idx, f"Completed task {idx} of {total_tasks}")
        self.finished.emit("✓ All tasks have been successfully processed!")

    def stop(self):
        """Signal the thread to stop processing."""
        self.stopped = True


class LogWindow(QWidget):
    def __init__(self):
        super().__init__()
        script_path = os.path.abspath(__file__)
        ui_path = os.path.join(os.path.dirname(script_path), "..", "ui", "LogWindow.ui")
        self.ui = uic.loadUi(ui_path, self)
        self.ui.setWindowTitle("EEG-COMET Log")
        self.ui.progress_stop_button.clicked.connect(self.stop_process)
        self.worker_thread = None
        # Initially, there is no process running so disable the stop button.
        self.ui.progress_stop_button.setEnabled(False)

    def append_log(self, log, log_type='info'):
        """Append a log entry with date and time."""
        current_date = datetime.now().strftime("%d/%m/%y")
        current_time = datetime.now().strftime("%I:%M %p")
        if log_type == 'settings':
            separator = "*" * 50
            current_log_text = f"\n{separator}\n{log}\n{separator}\n"
        else:
            current_log_text = f"[{current_date} {current_time}]: {log}\n"
        self.ui.log_text_area.append(current_log_text)

    def replace_log(self, log_text):
        self.ui.log_text_area.setText(log_text)

    def setup_progress_dialog(self, window_title, label_text, tasks, processing_func):
        """
        Set up and start the processing thread.

        param tasks: For example: list of tuples [(eeg_path, eeg_name), ...]
        param processing_func: A callable that does the heavy processing for a single file.
        """
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.show()
        self.setWindowTitle(window_title)
        self.ui.progress_label.setText(label_text)
        self.ui.progress_bar.setValue(0)

        if isinstance(tasks, int):
            total_tasks = tasks
        elif hasattr(tasks, '__len__') and not isinstance(tasks, str):
            total_tasks = len(tasks)
        else:
            total_tasks = 1
        self.ui.progress_bar.setRange(0, total_tasks)

        self.worker_thread = Worker(tasks, processing_func)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.finished.connect(self.process_finished)
        self.worker_thread.start()

    def update_progress(self, value, text):
        self.ui.progress_stop_button.setEnabled(True)
        """Update the progress bar and label."""
        self.ui.progress_bar.setValue(value)
        self.ui.progress_lineedit.setText(text)
        # Process pending events so the UI stays responsive.
        QApplication.processEvents()

    def process_finished(self, text=None):
        """Called when processing is finished."""
        self.setWindowFlags(Qt.Window)
        self.show()
        if text is not None:
            self.ui.progress_lineedit.setText(text)
        # Disable the stop button when processing is done.
        self.ui.progress_stop_button.setEnabled(False)

    def set_window_title(self, title):
        self.setWindowTitle(title)

    def set_label_text(self, text):
        self.ui.progress_label.setText(text)

    def stop_process(self):
        """Stop the process if it is running."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            # Optionally wait for the thread to finish.
            self.worker_thread.wait()
        self.ui.progress_stop_button.setEnabled(False)

    def show_hide_log_window(self):
        """Toggle the visibility of the log window."""
        self.setVisible(not self.isVisible())
