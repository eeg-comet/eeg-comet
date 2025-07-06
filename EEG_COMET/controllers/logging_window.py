import os.path
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMetaObject, Q_ARG, QCoreApplication
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
        self.tasks = tasks
        self.processing_func = processing_func
        self.stopped = False
        self.total_tasks = len(tasks) if hasattr(tasks, '__len__') else 1
        self.dynamic_total = None  # For dynamic task counts like TAAHC

    def run(self):
        total_tasks = len(self.tasks)
        for idx, task in enumerate(self.tasks, start=1):
            if self.stopped:
                self.finished.emit("Process stopped by user!")
                return

            if isinstance(task, (tuple, list)) and not isinstance(task, str):
                self.processing_func(*task)
            else:
                self.processing_func(task)

            # For standard progress (non-TAAHC)
            if self.dynamic_total is None:
                self.progress_updated.emit(idx, f"Completed task {idx} of {total_tasks}")

        self.finished.emit("✅ All tasks have been successfully processed!")

    def set_dynamic_total(self, total):
        """Set dynamic total for methods like TAAHC that have variable progress steps"""
        self.dynamic_total = total

    def stop(self):
        """Signal the thread to stop processing."""
        self.stopped = True


class LogWindow(QWidget):
    def __init__(self, comet_instance=None):
        super().__init__()
        script_path = os.path.abspath(__file__)
        ui_path = os.path.join(os.path.dirname(script_path), "..", "ui", "LogWindow.ui")
        self.ui = uic.loadUi(ui_path, self)
        self.ui.setWindowTitle("EEG-COMET Log")
        self.ui.progress_stop_button.clicked.connect(self.stop_process)
        self.worker_thread = None
        # Initially, there is no process running so disable the stop button.
        self.ui.progress_stop_button.setEnabled(False)
        
        # Store reference to COMET instance for log persistence
        self.comet_instance = comet_instance
        
        # Callback function to be called when processing is finished
        self.process_finished_callback = None

    def _is_main_thread(self):
        """Check if we're currently in the main thread."""
        return QCoreApplication.instance().thread() == QThread.currentThread()

    def append_log(self, log, log_type='info'):
        """Append a log entry with date and time."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if log_type == 'settings':
            separator = "=" * 60
            current_log_text = f"\n{separator}\n📋 CONFIGURATION SETTINGS\n{separator}\n{log}\n{separator}\n"
        elif log_type == 'success':
            current_log_text = f"[{current_date} {current_time}]\n✅ SUCCESS: {log}\n"
        elif log_type == 'error':
            current_log_text = f"[{current_date} {current_time}]\n❌ ERROR: {log}\n"
        elif log_type == 'warning':
            current_log_text = f"[{current_date} {current_time}]\n⚠️ WARNING: {log}\n"
        elif log_type == 'info':
            current_log_text = f"[{current_date} {current_time}]\nℹ️ INFO: {log}\n"
        elif log_type == 'process':
            current_log_text = f"[{current_date} {current_time}]\n🔄 PROCESSING: {log}\n"
        elif log_type == 'file':
            current_log_text = f"[{current_date} {current_time}]\n📁 FILE: {log}\n"
        elif log_type == 'section':
            separator = "-" * 50
            current_log_text = f"\n{separator}\n🔧 {log.upper()}\n{separator}\n"
        else:
            current_log_text = f"[{current_date} {current_time}]\n{log}\n"
        
        # Use thread-safe update only if we're not in the main thread
        if self._is_main_thread():
            self.ui.log_text_area.append(current_log_text)
        else:
            QMetaObject.invokeMethod(self.ui.log_text_area, "append", 
                                    Qt.QueuedConnection, Q_ARG(str, current_log_text))
        
        # Save log to COMET instance if available
        self._save_log_to_comet()

    def replace_log(self, log_text):
        """Replace the entire log content."""
        # Always use direct method for replace_log as it's typically called from main thread
        self.ui.log_text_area.setText(log_text)
        
        # Save log to COMET instance if available
        self._save_log_to_comet()

    def get_log_content(self):
        """Get the current log content as a string."""
        return self.ui.log_text_area.toPlainText()

    def set_log_content(self, log_content):
        """Set the log content from a string."""
        if log_content:
            # Ensure proper encoding handling for Unicode characters
            if isinstance(log_content, str):
                try:
                    # Handle potential encoding issues and ensure proper Unicode display
                    clean_content = log_content.encode('utf-8', errors='replace').decode('utf-8')
                    # Always use direct method for set_log_content as it's typically called from main thread
                    self.ui.log_text_area.setText(clean_content)
                except Exception:
                    # Fallback to original content
                    self.ui.log_text_area.setText(log_content)
            else:
                self.ui.log_text_area.setText(str(log_content))
                
            # Save log to COMET instance if available
            self._save_log_to_comet()

    def _save_log_to_comet(self):
        """Save the current log content to the COMET instance."""
        if self.comet_instance is not None:
            log_content = self.get_log_content()
            self.comet_instance.log_text = log_content

    def setup_progress_dialog(self, window_title, label_text, tasks, processing_func):
        """
        Set up and start the processing thread with enhanced progress tracking.

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

        # Set initial range - this may be updated for TAAHC
        self.ui.progress_bar.setRange(0, total_tasks)

        self.worker_thread = Worker(tasks, processing_func)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.finished.connect(self.process_finished)
        self.worker_thread.start()

    def update_progress(self, value, text):
        """Update the progress bar and label with dynamic range support."""
        self.ui.progress_stop_button.setEnabled(True)

        # Update progress bar maximum if needed (for TAAHC dynamic progress)
        if value > self.ui.progress_bar.maximum():
            self.ui.progress_bar.setMaximum(value * 2)  # Give some buffer

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
        
        # Call the callback function if it's set
        if self.process_finished_callback is not None:
            self.process_finished_callback()

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

    def clear_logs(self):
        """Clear all log content."""
        self.ui.log_text_area.clear()
        self._save_log_to_comet()

    def export_logs(self, file_path):
        """Export logs to a file."""
        log_content = self.get_log_content()
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
            return True
        except Exception as e:
            print(f"Error exporting logs: {e}")
            return False
