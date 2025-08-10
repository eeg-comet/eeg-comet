"""Logging window and background worker for EEG-COMET GUI."""

import os.path
from datetime import datetime

from PyQt5 import uic
from PyQt5.QtCore import Q_ARG, QCoreApplication, QMetaObject, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget


class Worker(QThread):
    """Background worker for step-based or task-based processing.

    Emits progress and finished signals during execution. Supports two modes:
    step-based (when ``tasks`` is an ``int``) and task-based (when ``tasks`` is
    an iterable of work items).

    Signals:
      progress_updated (pyqtSignal): Emitted with (int value, str text).
      finished (pyqtSignal): Emitted with a completion text.
    """
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(str)

    def __init__(self, tasks, processing_func):
        """Initialize the worker.

        Args:
          tasks: Either an ``int`` for total steps (step-based) or an iterable
            of tasks (task-based). Each task can be a single value or a tuple/list
            that will be unpacked for the processing function.
          processing_func (Callable): Function that processes a task or controls
            step-based processing. It may optionally accept a ``worker=`` kwarg.

        Returns:
          None
        """
        super().__init__()
        self.tasks = tasks
        self.processing_func = processing_func
        self.stopped = False
        # Handle both list of tasks and integer total steps
        if isinstance(tasks, int):
            self.total_tasks = tasks
            self.is_step_based = True
        else:
            self.total_tasks = len(tasks) if hasattr(tasks, "__len__") else 1
            self.is_step_based = False
        self.dynamic_total = None

    def run(self):
        """Execute the worker loop for step-based or task-based processing.

        Returns:
          None
        """
        if self.is_step_based:
            # For step-based processing (like clustering), just call the processing function once
            # The progress updates are handled within the processing function
            try:
                self.processing_func("step_based_processing", worker=self)
            except TypeError:
                # If the function doesn't accept worker parameter, call without it
                self.processing_func("step_based_processing")

            self.finished.emit("✅ Processing completed successfully!")
        else:
            # Original task-based processing
            total_tasks = len(self.tasks)
            for idx, task in enumerate(self.tasks, start=1):
                if self.stopped:
                    self.finished.emit("Process stopped by user!")
                    return

                # Pass the worker instance to allow checking stopped flag
                # Use try-except to handle functions that don't accept worker parameter
                try:
                    if isinstance(task, (tuple, list)) and not isinstance(task, str):
                        self.processing_func(*task, worker=self)
                    else:
                        self.processing_func(task, worker=self)
                except TypeError:
                    # If the function doesn't accept worker parameter, call without it
                    if isinstance(task, (tuple, list)) and not isinstance(task, str):
                        self.processing_func(*task)
                    else:
                        self.processing_func(task)

                # Only update progress when a task is actually completed
                # For clustering, this means each repetition is finished
                if self.dynamic_total is None:
                    self.progress_updated.emit(idx, f"Completed task {idx} of {total_tasks}")

            self.finished.emit("✅ All tasks have been successfully processed!")

    def stop(self):
        """Signal the thread to stop processing.

        Returns:
          None
        """
        self.stopped = True


class LogWindow(QWidget):
    """Log window with progress UI and utilities for long-running steps.

    Attributes:
      comet_instance: Optional COMET instance used for log persistence.
      worker_thread (Worker | None): Active background worker or None.
      process_finished_callback (Callable | None): Callback executed after finish.
      current_optimizer: Reference to current optimizer for stop functionality.
      current_step (str | None): Current processing step key.
    """

    def __init__(self, comet_instance=None):
        """Set up the log window UI and initialize state.

        Args:
          comet_instance: Optional COMET instance used to persist logs.

        Returns:
          None
        """
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

        # Re-entrancy guard to avoid recursive logging during save
        self._is_saving_logs = False

        # Callback function to be called when processing is finished
        self.process_finished_callback = None

        # Reference to current optimizer for stop functionality
        self.current_optimizer = None

        # Track current processing step for stop logging
        self.current_step = None

    @staticmethod
    def _is_main_thread():
        """Return True if currently in the main Qt thread.

        Returns:
          bool: True if in main thread; otherwise False.
        """
        return QCoreApplication.instance().thread() == QThread.currentThread()

    def append_log(self, log, log_type="info"):
        """Append a log entry with timestamp to the UI and persist to COMET.

        Args:
          log (str): Log message body.
          log_type (str): One of {"settings", "success", "error", "warning",
            "info", "process", "file", "section"}.

        Returns:
          None
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        if log_type == "settings":
            separator = "=" * 60
            current_log_text = (
                f"\n{separator}\n📋 CONFIGURATION SETTINGS\n{separator}\n{log}\n{separator}\n"
            )
        elif log_type == "success":
            current_log_text = f"[{current_date} {current_time}]\n✅ SUCCESS: {log}\n"
        elif log_type == "error":
            current_log_text = f"[{current_date} {current_time}]\n❌ ERROR: {log}\n"
        elif log_type == "warning":
            current_log_text = f"[{current_date} {current_time}]\n⚠️ WARNING: {log}\n"
        elif log_type == "info":
            current_log_text = f"[{current_date} {current_time}]\nℹ️ INFO: {log}\n"
        elif log_type == "process":
            current_log_text = f"[{current_date} {current_time}]\n🔄 PROCESSING: {log}\n"
        elif log_type == "file":
            current_log_text = f"[{current_date} {current_time}]\n📁 FILE: {log}\n"
        elif log_type == "section":
            separator = "-" * 50
            current_log_text = f"\n{separator}\n🔧 {log.upper()}\n{separator}\n"
        else:
            current_log_text = f"[{current_date} {current_time}]\n{log}\n"

        # Use thread-safe update only if we're not in the main thread
        if self._is_main_thread():
            self.ui.log_text_area.append(current_log_text)
        else:
            QMetaObject.invokeMethod(
                self.ui.log_text_area, "append", Qt.QueuedConnection, Q_ARG(str, current_log_text)
            )

        # Save log to COMET instance if available
        self._save_log_to_comet()

    def replace_log(self, log_text):
        """Replace the entire log content.

        Args:
          log_text (str): New full log text.

        Returns:
          None
        """
        # Always use direct method for replace_log as it's typically called from main thread
        self.ui.log_text_area.setText(log_text)

        # Save log to COMET instance if available
        self._save_log_to_comet()

    def get_log_content(self):
        """Get the current log content as a string.

        Returns:
          str: Full log content.
        """
        return self.ui.log_text_area.toPlainText()

    def set_log_content(self, log_content):
        """Set the log content from a string, handling Unicode gracefully.

        Args:
          log_content (str): New content to display in the log window.

        Returns:
          None
        """
        if log_content:
            # Ensure proper encoding handling for Unicode characters
            if isinstance(log_content, str):
                try:
                    # Handle potential encoding issues and ensure proper Unicode display
                    clean_content = log_content.encode("utf-8", errors="replace").decode("utf-8")
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
        """Save the current log content to the COMET instance, if available, without recursion."""
        if self.comet_instance is None or self._is_saving_logs:
            return
        self._is_saving_logs = True
        try:
            log_content = self.get_log_content()
            self.comet_instance.log_text = log_content
            # Also save to the separate log file
            self.comet_instance.save_logs_to_file()
        finally:
            self._is_saving_logs = False

    def setup_progress_dialog(self, window_title, label_text, tasks, processing_func):
        """Set up and start the processing thread with enhanced progress tracking.

        Args:
          window_title (str): Title for the progress window.
          label_text (str): Label text to describe the current operation.
          tasks: Either an int (repetitions/steps) or an iterable of tasks.
          processing_func (Callable): Function to process a task or drive steps.

        Returns:
          None
        """
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.show()
        self.setWindowTitle(window_title)
        self.ui.progress_label.setText(label_text)
        self.ui.progress_bar.setValue(0)

        # Determine current step from window title
        if "Preprocessing" in window_title:
            self.current_step = "PREPROCESSING"
        elif "Clustering" in window_title:
            self.current_step = "CLUSTERING"
            # Log clustering start with setup information
            if isinstance(tasks, int):
                self.append_log(
                    f"🚀 Starting microstate clustering with {tasks} repetitions",
                    log_type="section",
                )
                self.append_log("📋 Initializing clustering process...", log_type="info")
        elif "Backfitting" in window_title:
            self.current_step = "BACKFITTING"
        elif "Feature" in window_title:
            self.current_step = "FEATURE_EXTRACTION"
        elif "Source" in window_title:
            self.current_step = "SOURCE_LOCALIZATION"
        else:
            self.current_step = "PROCESSING"

        if isinstance(tasks, int):
            total_tasks = tasks
        elif hasattr(tasks, "__len__") and not isinstance(tasks, str):
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
        """Update the progress bar and label with dynamic range support.

        Args:
          value (int): Progress value to display.
          text (str): Descriptive progress text.

        Returns:
          None
        """
        self.ui.progress_stop_button.setEnabled(True)

        # Update progress bar maximum if needed (for TAAHC dynamic progress)
        if value > self.ui.progress_bar.maximum():
            self.ui.progress_bar.setMaximum(value * 2)  # Give some buffer

        self.ui.progress_bar.setValue(value)
        self.ui.progress_lineedit.setText(text)

        # Add real-time logging for clustering progress
        if self.current_step == "CLUSTERING" and "completed" in text.lower():
            # Extract repetition number from text
            if "repetition" in text.lower():
                try:
                    # Parse repetition number from text like "Clustering repetition 1/5 completed"
                    parts = text.split()
                    rep_idx = parts.index("repetition") + 1
                    rep_num = parts[rep_idx].split("/")[0]
                    total_reps = parts[rep_idx].split("/")[1]

                    # Calculate percentage
                    percentage = int((int(rep_num) / int(total_reps)) * 100)

                    # Log the completion with more details
                    log_message = (
                        f"✅ Clustering repetition {rep_num}/{total_reps} completed ({percentage}%)"
                    )
                    self.append_log(log_message, log_type="success")

                except (IndexError, ValueError):
                    # Fallback if parsing fails
                    self.append_log(f"✅ {text}", log_type="success")
            elif "taahc" in text.lower():
                self.append_log(f"✅ {text}", log_type="success")

        # Process pending events so the UI stays responsive.
        QApplication.processEvents()

    def process_finished(self, text=None):
        """Handle completion of processing and finalize UI/logs.

        Args:
          text (str | None): Final status text to display; optional.

        Returns:
          None
        """
        self.setWindowFlags(Qt.Window)
        self.show()
        if text is not None:
            self.ui.progress_lineedit.setText(text)

            # Log completion based on current step
            if self.current_step == "CLUSTERING":
                if "successfully" in text.lower():
                    self.append_log(
                        "🎉 Clustering process completed successfully!", log_type="success"
                    )
                    self.append_log("📊 Results saved and ready for analysis", log_type="info")
                else:
                    self.append_log(f"⚠️ Clustering process finished: {text}", log_type="warning")

        # Disable the stop button when processing is done.
        self.ui.progress_stop_button.setEnabled(False)

        # Call the callback function if it's set
        if self.process_finished_callback is not None:
            self.process_finished_callback()

    def set_window_title(self, title):
        """Set the window title text.

        Args:
          title (str): New title for the log window.

        Returns:
          None
        """
        self.setWindowTitle(title)

    def set_label_text(self, text):
        """Set the progress label text.

        Args:
          text (str): Label text describing the current operation.

        Returns:
          None
        """
        self.ui.progress_label.setText(text)

    def stop_process(self):
        """Stop the process if it is running.

        Returns:
          None
        """
        # Log stop request if we have a current step
        if self.current_step and self.comet_instance and hasattr(self.comet_instance, "logger"):
            self.comet_instance.logger.stop_requested(self.current_step)

        # Log stop action in the log window
        if self.current_step == "CLUSTERING":
            self.append_log("⏹️ Clustering process stopped by user", log_type="warning")
            self.append_log("💾 Partial results will be saved if available", log_type="info")

        # Stop the worker thread if running
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            # Optionally wait for the thread to finish.
            self.worker_thread.wait()

        # Stop the optimizer if it's running
        if hasattr(self, "current_optimizer") and self.current_optimizer is not None:
            self.current_optimizer.stop()
            self.current_optimizer = None

        self.ui.progress_stop_button.setEnabled(False)

    def show_hide_log_window(self):
        """Toggle the visibility of the log window.

        Returns:
          None
        """
        self.setVisible(not self.isVisible())

    def clear_logs(self):
        """Clear all log content and persist the cleared state.

        Returns:
          None
        """
        self.ui.log_text_area.clear()
        self._save_log_to_comet()

    def export_logs(self, file_path):
        """Export logs to a file.

        Args:
          file_path (str): Destination file path.

        Returns:
          bool: True if export succeeded; otherwise False.
        """
        log_content = self.get_log_content()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(log_content)
            return True
        except Exception as e:
            print(f"Error exporting logs: {e}")
            return False

    def closeEvent(self, event):
        """Handle window close event and persist final logs.

        Args:
          event: Qt close event.

        Returns:
          None
        """
        # Add closing log before closing
        self.append_log("EEG-COMET Session Ended", log_type="section")
        self.append_log("Thank you for using EEG-COMET!", log_type="info")

        # Save final log state
        self._save_log_to_comet()

        # Accept the close event
        event.accept()

    def log_clustering_setup_step(self, step_name, step_description=""):
        """Log clustering setup steps for better user visibility.

        Args:
          step_name (str): Short step identifier.
          step_description (str): Optional detailed description.

        Returns:
          None
        """
        if self.current_step == "CLUSTERING":
            # Use ⌛ for actual processing steps that take time
            if step_name.lower() == "starting clustering":
                if step_description:
                    self.append_log(f"⌛ {step_name}: {step_description}", log_type="process")
                else:
                    self.append_log(f"⌛ {step_name}", log_type="process")
            else:
                # Use ⚙️ for setup steps
                if step_description:
                    self.append_log(f"⚙️ {step_name}: {step_description}", log_type="process")
                else:
                    self.append_log(f"⚙️ {step_name}", log_type="process")

    def log_clustering_progress(self, repetition_num, total_repetitions, gev=None, is_best=False):
        """Log detailed clustering progress information.

        Args:
          repetition_num (int): Current repetition index (1-based).
          total_repetitions (int): Total number of repetitions.
          gev (float | None): Optional GEV value for this repetition.
          is_best (bool): Whether the GEV is the best so far.

        Returns:
          None
        """
        if self.current_step == "CLUSTERING":
            percentage = int((repetition_num / total_repetitions) * 100)
            base_message = (
                f"🔄 Clustering repetition {repetition_num}/{total_repetitions} ({percentage}%)"
            )

            if gev is not None:
                gev_percent = f"{gev * 100:.2f}%"
                if is_best:
                    self.append_log(
                        f"{base_message} - GEV: {gev_percent} (Best so far!)", log_type="success"
                    )
                else:
                    self.append_log(f"{base_message} - GEV: {gev_percent}", log_type="info")
            else:
                self.append_log(base_message, log_type="info")
