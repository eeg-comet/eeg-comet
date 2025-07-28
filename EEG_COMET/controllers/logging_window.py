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
        param tasks: A list of tasks or an integer representing total steps. Each task can be any structure (e.g., a tuple, list, or single value)
                      that processing_func can handle.
        param processing_func: A callable that processes a task. It should accept the task (or unpacked values
                                if the task is a tuple or list).
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
            self.total_tasks = len(tasks) if hasattr(tasks, '__len__') else 1
            self.is_step_based = False
        self.dynamic_total = None

    def run(self):
        if self.is_step_based:
            # For step-based processing (like clustering), just call the processing function once
            # The progress updates are handled within the processing function
            try:
                self.processing_func('step_based_processing', worker=self)
            except TypeError:
                # If the function doesn't accept worker parameter, call without it
                self.processing_func('step_based_processing')
            
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
        
        # Reference to current optimizer for stop functionality
        self.current_optimizer = None
        
        # Track current processing step for stop logging
        self.current_step = None

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
            # Also save to the separate log file
            self.comet_instance.save_logs_to_file()

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

        # Determine current step from window title
        if "Preprocessing" in window_title:
            self.current_step = "PREPROCESSING"
        elif "Clustering" in window_title:
            self.current_step = "CLUSTERING"
            # Log clustering start with setup information
            if isinstance(tasks, int):
                self.append_log(f"🚀 Starting microstate clustering with {tasks} repetitions", log_type='section')
                self.append_log("📋 Initializing clustering process...", log_type='info')
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
        
        # Add real-time logging for clustering progress
        if self.current_step == "CLUSTERING" and "completed" in text.lower():
            # Extract repetition number from text
            if "repetition" in text.lower():
                try:
                    # Parse repetition number from text like "Clustering repetition 1/5 completed"
                    parts = text.split()
                    rep_idx = parts.index("repetition") + 1
                    rep_num = parts[rep_idx].split('/')[0]
                    total_reps = parts[rep_idx].split('/')[1]
                    
                    # Calculate percentage
                    percentage = int((int(rep_num) / int(total_reps)) * 100)
                    
                    # Log the completion with more details
                    log_message = f"✅ Clustering repetition {rep_num}/{total_reps} completed ({percentage}%)"
                    self.append_log(log_message, log_type='success')
                    
                except (IndexError, ValueError):
                    # Fallback if parsing fails
                    self.append_log(f"✅ {text}", log_type='success')
            elif "taahc" in text.lower():
                self.append_log(f"✅ {text}", log_type='success')
        
        # Process pending events so the UI stays responsive.
        QApplication.processEvents()

    def process_finished(self, text=None):
        """Called when processing is finished."""
        self.setWindowFlags(Qt.Window)
        self.show()
        if text is not None:
            self.ui.progress_lineedit.setText(text)
            
            # Log completion based on current step
            if self.current_step == "CLUSTERING":
                if "successfully" in text.lower():
                    self.append_log("🎉 Clustering process completed successfully!", log_type='success')
                    self.append_log("📊 Results saved and ready for analysis", log_type='info')
                else:
                    self.append_log(f"⚠️ Clustering process finished: {text}", log_type='warning')
        
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
        # Log stop request if we have a current step
        if self.current_step and self.comet_instance and hasattr(self.comet_instance, 'logger'):
            self.comet_instance.logger.stop_requested(self.current_step)
        
        # Log stop action in the log window
        if self.current_step == "CLUSTERING":
            self.append_log("⏹️ Clustering process stopped by user", log_type='warning')
            self.append_log("💾 Partial results will be saved if available", log_type='info')
        
        # Stop the worker thread if running
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            # Optionally wait for the thread to finish.
            self.worker_thread.wait()
        
        # Stop the optimizer if it's running
        if hasattr(self, 'current_optimizer') and self.current_optimizer is not None:
            self.current_optimizer.stop()
            self.current_optimizer = None
        
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

    def closeEvent(self, event):
        """Handle window close event"""
        # Add closing log before closing
        self.append_log("EEG-COMET Session Ended", log_type='section')
        self.append_log("Thank you for using EEG-COMET!", log_type='info')
        
        # Save final log state
        self._save_log_to_comet()
        
        # Accept the close event
        event.accept()

    def log_clustering_setup_step(self, step_name, step_description=""):
        """Log clustering setup steps for better user visibility."""
        if self.current_step == "CLUSTERING":
            # Use ⌛ for actual processing steps that take time
            if step_name.lower() == "starting clustering":
                if step_description:
                    self.append_log(f"⌛ {step_name}: {step_description}", log_type='process')
                else:
                    self.append_log(f"⌛ {step_name}", log_type='process')
            else:
                # Use ⚙️ for setup steps
                if step_description:
                    self.append_log(f"⚙️ {step_name}: {step_description}", log_type='process')
                else:
                    self.append_log(f"⚙️ {step_name}", log_type='process')

    def log_clustering_progress(self, repetition_num, total_repetitions, gev=None, is_best=False):
        """Log detailed clustering progress information."""
        if self.current_step == "CLUSTERING":
            percentage = int((repetition_num / total_repetitions) * 100)
            base_message = f"🔄 Clustering repetition {repetition_num}/{total_repetitions} ({percentage}%)"
            
            if gev is not None:
                gev_percent = f"{gev * 100:.2f}%"
                if is_best:
                    self.append_log(f"{base_message} - GEV: {gev_percent} (Best so far!)", log_type='success')
                else:
                    self.append_log(f"{base_message} - GEV: {gev_percent}", log_type='info')
            else:
                self.append_log(base_message, log_type='info')
