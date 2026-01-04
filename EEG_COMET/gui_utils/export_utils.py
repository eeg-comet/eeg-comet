"""Utilities for consistent file export dialogs and matplotlib figure saving."""

from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QWidget


def get_save_file_path(
    parent: QWidget,
    default_name: str,
    dialog_title: str,
    name_filter: str = "PDF Files (*.pdf);;PNG Files (*.png);;JPG Files (*.jpg);;SVG Files (*.svg);;All Files (*)",
) -> Optional[str]:
    """Open a save-file dialog and return the chosen file path, or None if cancelled."""
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getSaveFileName(
        parent, dialog_title, default_name, name_filter, options=options
    )
    return file_name or None


def save_matplotlib_figure(
    *,
    figure,
    canvas,
    file_name: str,
    title_text: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    font_family: Optional[str] = None,
) -> None:
    """Save a matplotlib figure with sensible defaults for vector/raster formats.

    Adds an optional temporary title, saves with format-appropriate settings,
    then removes the title and redraws the canvas.
    """
    # Ensure extension; default to PDF for vector format
    extension = os.path.splitext(file_name)[-1].lower()
    if not extension:
        file_name += ".pdf"
        extension = ".pdf"

    # Add a temporary title if provided
    if title_text:
        figure.suptitle(
            title_text,
            fontsize=(title_fontsize or 16),
            fontweight="bold",
            fontfamily=(font_family or "sans-serif"),
        )

    # Save with appropriate settings
    if extension in [".pdf", ".svg"]:
        canvas.figure.savefig(
            file_name,
            format=extension[1:],
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    else:
        canvas.figure.savefig(file_name, dpi=300, bbox_inches="tight")

    # Remove the temporary title and refresh
    if title_text:
        figure.suptitle("")
    canvas.draw()


