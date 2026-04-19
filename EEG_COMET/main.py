"""EEG-COMET GUI application entrypoint."""

import logging
import os
import sys

import mne
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

# Enable high-DPI scaling BEFORE creating QApplication
# This must be done before any QApplication instance is created
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from controllers.main_microstate_window import MainMicrostateWindow
from gui_utils.terminal_logger import get_logger
from gui_utils.ui_scale import UIScale

# Set EEG_COMET_SHOW_MONTAGE_WARNINGS=1 to surface every electrode-position
# warning emitted by MNE; otherwise these messages are suppressed and the
# first occurrence is logged once.
mne.set_log_level("ERROR")


class MNEWarningFilter(logging.Filter):
    """Filter MNE electrode-position warnings; emit each unique warning once."""

    _SUPPRESSED_NEEDLES = (
        "Did not find any electrode locations",
        "digitization points do not correspond",
    )

    def __init__(self):
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record):
        try:
            message = record.getMessage() if hasattr(record, "getMessage") else ""
            for needle in self._SUPPRESSED_NEEDLES:
                if needle in message:
                    if needle not in self._seen:
                        self._seen.add(needle)
                        logging.getLogger("eeg_comet.mne_warnings").warning(
                            "Suppressing repeated MNE warning: %r", needle
                        )
                    return False
        except Exception:
            return True
        return True


if os.environ.get("EEG_COMET_SHOW_MONTAGE_WARNINGS", "").lower() not in ("1", "true", "yes"):
    # Attach the filter to the MNE logger only so unrelated libraries are
    # not affected by this suppression.
    logging.getLogger("mne").addFilter(MNEWarningFilter())

# Initialize global logger
logger = get_logger()


class CustomApplicationContext:
    """Custom application context for EEG-COMET."""

    def __init__(self):
        """Create the QApplication and resolve base path for resources."""
        self.app = QApplication([])
        self.base_path = os.path.dirname(__file__)
        self.base_qss_template = self._load_base_qss()

    def get_resource(self, path):
        """Return the full path of a resource file."""
        return os.path.join(self.base_path, "ui", path)

    def _load_base_qss(self):
        """Read ``ui/theme.qss`` and return its contents (or ``""`` on miss)."""
        qss_path = os.path.join(self.base_path, "ui", "theme.qss")
        if not os.path.exists(qss_path):
            return ""
        try:
            with open(qss_path, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError as exc:
            logger.warning("STARTUP", f"Could not load theme.qss: {exc}")
            return ""


def run_application():
    """Run the EEG-COMET application."""
    # Display welcome message
    logger.toolbox_header("EEG-COMET", "(EEG Comprehensive Microstate Extraction Toolbox)")
    logger.processing_info("ORGANIZATION", "SFU eBrain Lab")
    logger.processing_info("CONTACT", "     https://www.ebrainlab.ca/about-us")
    logger.processing_info("GITHUB", "      https://github.com/eBrainLab/eeg-comet")
    logger.processing_info("MAINTENANCE", " https://github.com/aminkabir")

    try:
        # Create a custom application context
        app_context = CustomApplicationContext()
        app = app_context.app

        # Initialize main window
        window = MainMicrostateWindow(app_context, parent=None)

        # Now that every widget has been built, install the global UI-scale
        # controller and let the main window register its menu actions.
        ui_scale = UIScale(
            app,
            base_qss_template=app_context.base_qss_template,
            base_font_pt=getattr(window, "_base_font_pt", 14),
            font_family=getattr(window, "_font_family", "Calibri"),
            theme_qss=getattr(window, "light_style", ""),
        )
        window.attach_ui_scale(ui_scale)
        ui_scale.apply()

        # Set the application icon
        icon_path = app_context.get_resource("eeg_comet_logo.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        else:
            logger.warning("STARTUP", "Application icon not found")

        # Show the window
        window.show()
        logger.processing_success("STARTUP", "     EEG-COMET GUI Started Successfully")

        # Start the application
        exit_code = app.exec_()

        # NOTE: do NOT call back into ``window`` / ``window.comet.LogWindow``
        # here. The main window's closeEvent has already fired (that's why
        # exec_ returned), so reaching into Qt widgets at this point can hit
        # already-deleted C++ objects. The shutdown banner is emitted from
        # ``MainMicrostateWindow.closeEvent`` and ``LogWindow.closeEvent``.
        for widget in app.topLevelWidgets():
            try:
                if widget.isVisible():
                    widget.close()
            except RuntimeError:
                pass

        # Add separator before shutdown message
        print()  # Empty line
        print("=" * 60)  # Separator line

        logger.processing_info("SHUTDOWN", "    EEG-COMET GUI Closed Successfully")
        sys.exit(exit_code)

    except Exception as e:
        logger.error("STARTUP", f"Failed to Start EEG-COMET GUI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_application()
