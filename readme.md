# ☄️ EEG-COMET — EEG Comprehensive Microstate Extraction Toolbox

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Platform: Windows | macOS | Linux](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](#installation)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](#)

EEG-COMET is an open-source, end-to-end toolbox for **EEG microstate analysis**, developed by the [SFU eBrain Lab](https://www.ebrainlab.ca). It unifies preprocessing, automated cluster validation, machine-learning–based labeling, backfitting, feature extraction, and source localization in one reproducible workflow, accessible from either a GUI or a command-line interface.

> Full documentation: **<[https://eeg-comet.github.io](https://eeg-comet.github.io/eeg-comet/)>**

---

## Highlights

- **End-to-end pipeline** — quality control, preprocessing, clustering, labeling, backfitting, features, statistics, and source localization in a single tool.
- **Three analysis modes** — full-recording, sliding-window dynamic, and single-trial event-related microstate analysis.
- **ML-based labeling** — objective assignment to canonical microstate classes via a built-in CNN, eliminating subjective visual inspection.
- **Automated cluster validation** — 10 metrics with multiple selection strategies for choosing the optimal number of microstates.
- **Source localization** — cortical estimation with dSPM, MNE, sLORETA, and eLORETA on standardized or individual anatomy.
- **Reproducible runs** — every session writes a structured log; configurations round-trip through INI files.
- **GUI and CLI** — point-and-click for new users, scriptable for batch processing.

## Requirements

- Python 3.10 – 3.12
- A working Qt platform (PyQt5 is installed automatically; on Linux you may need system OpenGL libraries for PyVista)
- See [`requirements.txt`](requirements.txt) for fully pinned dependencies, or [`environment.yml`](environment.yml) for the conda recipe.

## Installation

Clone the repository, then choose **one** of the two environments below.

```bash
git clone https://github.com/eeg-comet/eeg-comet.git
cd eeg-comet
```

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate eegcomet
```

### Option B — Python venv + pip

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Developing on EEG-COMET?** Use `pip install -e ".[dev]"` from the repo root to install the package in editable mode together with the test/lint extras declared in [`pyproject.toml`](pyproject.toml).

## Usage

### Graphical interface

```bash
cd EEG_COMET
python main.py
```

After installation as a package, the same GUI is available as a console script:

```bash
eeg-comet
```

### Command-line interface

```bash
eeg-comet-cli --help
```

The CLI consumes the same configuration schema as the GUI; see the [Parameters Reference](https://eeg-comet.github.io/parameters.html) for all options.

### First steps

1. Launch the GUI and create a new study from your preprocessed EEG files (`.set`, `.fif`, `.edf`, BIDS, …).
2. Select data, validate the cluster count, run clustering, and label microstates.
3. Backfit, extract features, and (optionally) source-localize the templates.

A full walkthrough lives in the [Getting Started guide](https://eeg-comet.github.io/getting-started.html).

> **Important.** Apply proper artifact rejection — particularly ocular artifacts — *before* importing data into EEG-COMET. Microstate analysis is highly sensitive to residual artifacts.

## Project layout

```
EEG_COMET/                 # Application package
├── controllers/           # PyQt5 window controllers
├── ui/                    # Qt Designer .ui files and theme.qss
├── clustering_utils/      # Modified K-means, TAAHC, optimizer
├── backfitting_utils/     # Backfitting and segmentation
├── features_utils/        # Feature extraction
├── sourcelocalization_utils/
├── data_utils/            # I/O, manifests, study management
├── gui_utils/             # Logging, layout, helpers
├── pipeline/              # Headless run orchestration
├── models/                # Bundled ONNX classifier
├── main.py                # GUI entry point
└── terminal_version.py    # CLI entry point
docs/                      # Jekyll source for the documentation site
tests/                     # Pytest suite
```

## How to cite

If you use EEG-COMET in academic work, please cite the companion paper. A machine-readable [`CITATION.cff`](CITATION.cff) is provided at the repository root, and GitHub renders a "Cite this repository" button from it. As a fallback citation:

> Kabir, A., Tarailis, P., Chatterjee, R., Dhami, P., & Farzan, F. (2026). *EEG-COMET: A Comprehensive Platform for Resting-State and Single-Trial Event-Related Microstate Analysis.* https://github.com/eeg-comet/eeg-comet

Once the paper is published, the journal reference and DOI will be added to `CITATION.cff`; please cite the published paper from that point on.

## License

EEG-COMET is released under the **GNU General Public License v3.0** — see [`LICENSE`](LICENSE) for the full text.

In short:

- You are free to use, study, modify, and redistribute the toolbox.
- Any redistributed or modified version (including a fork or a tool that incorporates EEG-COMET) **must also be released under GPL-3.0** with the copyright and license notices intact.
- The software is provided **without warranty of any kind**, to the extent permitted by applicable law.

For closed-source or otherwise GPL-incompatible use, please contact the authors to discuss a separate commercial license.

## Support and contributing

- **Bug reports and feature requests:** [GitHub Issues](https://github.com/eeg-comet/eeg-comet/issues)
- **Questions and collaboration:** [www.ebrainlab.ca](https://www.ebrainlab.ca)
- **Pull requests** are welcome; please open an issue first to discuss substantial changes.

---

<sub>EEG-COMET is developed and maintained by the <a href="https://www.ebrainlab.ca">SFU eBrain Lab</a>, Simon Fraser University.</sub>
