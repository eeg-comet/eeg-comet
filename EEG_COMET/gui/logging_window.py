
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
from datetime import datetime


class LogWindow(QWidget):
    def __init__(self, context, main_window=None, comet_tbx=None):
        super().__init__()
        self.main_window = main_window
        self.comet_tbx = comet_tbx
        self.ui = uic.loadUi(context.get_resource("LogWindow.ui"), self)
        self.ui.setWindowTitle("EEG-COMET Log")

    def append_log(self, log):
        # Get the current date and time
        current_date = datetime.now().strftime("%d/%m/%y")
        current_time = datetime.now().strftime("%I:%M %p")

        # Format the log entry with date and time
        log_text = f"[{current_date} {current_time}]: {log}\n"

        # Append the log entry to the text area
        self.textArea.append(log_text)

    def replace_log(self, import_log):
        # Convert the list of log entries into a single string
        log_text = '\n'.join(import_log)
        # Replace the current log with the imported log
        self.textArea.setText(log_text)

    def show_hide_log_window(self):
        # Toggle the visibility of the log window
        if self.isVisible():
            self.setVisible(False)
        else:
            self.setVisible(True)