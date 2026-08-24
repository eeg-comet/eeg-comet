---
title: Installation
layout: default
nav_order: 2
description: "Step-by-step installation guide for EEG-COMET"
---

# Installation Guide
{: .no_toc }

Complete installation instructions for EEG-COMET on Windows, macOS, and Linux.
{: .fs-6 .fw-300 }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|:----------|:------------|
| **Operating System** | Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+) |
| **Python** | 3.10, 3.11, or 3.12 (`requires-python = ">=3.10,<3.13"`) |
| **RAM** | 8 GB minimum (16 GB recommended) |
| **Storage** | 5 GB for installation + space for data |
| **Display** | 1920×1080 resolution recommended |

### Recommended Specifications

For optimal performance, especially with large datasets or source localization:

- **RAM:** 16-32 GB
- **CPU:** Multi-core processor (4+ cores)
- **GPU:** Not required. Microstate classification runs on the CPU via [ONNX Runtime](https://onnxruntime.ai); no CUDA / cuDNN setup is necessary.

---

## Installation Methods

EEG-COMET can be installed using either Conda (recommended) or pip.

### Method 1: Conda Installation (Recommended)

Conda provides better dependency management and is the recommended installation method.

#### Prerequisites

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight) or [Anaconda](https://www.anaconda.com/products/distribution) (full distribution)

2. Verify installation:
   ```bash
   conda --version
   ```

#### Installation Steps

**Step 1: Clone the Repository**

```bash
git clone https://github.com/eeg-comet/eeg-comet.git
cd eeg-comet
```

{: .note }
> Alternatively, download the repository as a ZIP file from GitHub and extract it.

**Step 2: Create the Conda Environment**

```bash
conda env create -f environment.yml
```

This creates an environment named `eeg-comet` with all required dependencies, and installs the `eeg_comet` package itself (editable) so the `eeg-comet` / `eeg-comet-cli` entry points are registered.

**Step 3: Activate the Environment**

```bash
conda activate eeg-comet
```

**Step 4: Verify Installation**

```bash
python -c "import mne, sklearn, onnxruntime, PyQt5; print('Installation successful!')"
```

---

### Method 2: pip Installation

Use pip if you prefer Python's built-in virtual environment or don't have Conda installed.

#### Windows

```powershell
# Clone the repository
git clone https://github.com/eeg-comet/eeg-comet.git
cd eeg-comet

# Create virtual environment
python -m venv eeg-comet

# Activate the environment
.\eeg-comet\Scripts\Activate.ps1

# Install dependencies, then the package itself
pip install -r requirements.txt
pip install -e .
```

#### macOS / Linux

```bash
# Clone the repository
git clone https://github.com/eeg-comet/eeg-comet.git
cd eeg-comet

# Create virtual environment
python3 -m venv eeg-comet

# Activate the environment
source eeg-comet/bin/activate

# Install dependencies, then the package itself
pip install -r requirements.txt
pip install -e .
```

---

### Method 3: Editable / Developer Install

If you plan to modify EEG-COMET, install it in editable mode together with the development extras (tests, linters, build tools):

```bash
git clone https://github.com/eeg-comet/eeg-comet.git
cd eeg-comet
pip install -e ".[dev]"
```

This wires up the `eeg-comet` (GUI) and `eeg-comet-cli` (terminal) console scripts declared in `pyproject.toml`. See the [`pyproject.toml`](https://github.com/eeg-comet/eeg-comet/blob/stable/pyproject.toml) for available extras (`test`, `lint`, `docs`, `dev`, `all`).

---

## Launching EEG-COMET

After installation, launch the graphical interface using either of the supported entry points:

```bash
# 1) Console script (works after `pip install -e .` or `pip install eeg-comet`)
eeg-comet

# 2) Equivalent: run the package module directly
python -m eeg_comet.main
```

A terminal-only entry point is also available:

```bash
eeg-comet-cli --help
```

It always requires `--config`, plus at least one analysis step:

| Flag | Purpose |
|:-----|:--------|
| `--config PATH` | Path to the configuration file (required) |
| `--study`, `--input`, `--output` | Override the study name, input folder, and output folder from the config |
| `--preprocess`, `--cluster`, `--label`, `--backfit`, `--features`, `--source`, `--correlation` | Run individual analysis steps |
| `--all` | Run all analysis steps in sequence |
| `--auto-k`, `--k-min`, `--k-max` | Search automatically for the best number of clusters (defaults 2 and 10) |
| `--k` | Use a specific number of clusters (mutually exclusive with `--auto-k`) |
| `--method` | Clustering method: `kmeans`, `similarity`, or `taahc` |
| `--repeats` | Number of clustering repetitions |
| `--verbose` / `-v`, `--quiet` / `-q` | Increase or suppress output |

```bash
# Run the full pipeline with automatic k selection
eeg-comet-cli --config my_config.ini --all --auto-k --k-min 2 --k-max 8
```

{: .highlight }
> On first launch, EEG-COMET may take a few moments to load as it initializes the ONNX classification model and warms up Qt / pyvista resources.

---

## Key Dependencies

EEG-COMET relies on several major scientific Python packages. The versions below are the pins used in [`requirements.txt`](https://github.com/eeg-comet/eeg-comet/blob/stable/requirements.txt) for the reproducible install; `pyproject.toml` declares looser minimum bounds for PyPI installs.

| Package | Pinned version | Purpose |
|:--------|:---------------|:--------|
| `mne` | 1.8.0 | EEG data handling and processing |
| `mne-bids` | 0.16.0 | BIDS format support |
| `mne-qt-browser` | 0.6.3 | Interactive raw-data browser |
| `scikit-learn` | 1.5.0 | Clustering algorithms and validation |
| `onnxruntime` | 1.19.0 | CNN microstate-classification inference |
| `statsmodels` | 0.14.4 | Statistical analysis (GEE, LMM, multiple-comparison correction) |
| `pyvista` | 0.45.2 | 3D visualization for source localization |
| `pyvistaqt` | 0.11.2 | Qt embedding for pyvista plots |
| `PyQt5` | 5.15.11 | Graphical user interface |
| `QDarkStyle` | 3.2.3 | Application theming |
| `pandas` | 2.3.0 | Data manipulation and export |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `h5py` | 3.14.0 | HDF5 export support |

{: .note }
> EEG-COMET does **not** depend on TensorFlow. The microstate classifier is shipped as an ONNX file (`eeg_comet/models/model_v2.onnx`) and executed by `onnxruntime` on the CPU — no CUDA / cuDNN setup is required.

---

## Troubleshooting

### Common Issues

<div class="callout warning">
<strong>Qt Platform Plugin Error (Windows)</strong><br>
If you see "qt.qpa.plugin: Could not find the Qt platform plugin", try:
<pre>conda install -c conda-forge qt</pre>
</div>

<div class="callout warning">
<strong>ONNX Runtime fails to load the classifier</strong><br>
Make sure <code>eeg_comet/models/model_v2.onnx</code> exists in your install (it is bundled with the wheel and the source tarball). If it is missing, re-clone the repository or reinstall the package.
</div>

<div class="callout warning">
<strong>MNE-Qt-Browser Issues</strong><br>
If the data browser doesn't display correctly:
<pre>pip install --upgrade mne-qt-browser pyqtgraph</pre>
</div>

### Environment Conflicts

If you experience dependency conflicts:

```bash
# Remove existing environment
conda deactivate
conda env remove -n eeg-comet

# Recreate from scratch
conda env create -f environment.yml
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/eeg-comet/eeg-comet/issues) for similar problems
2. Create a new issue with:
   - Your operating system and version
   - Python version (`python --version`)
   - EEG-COMET version (`python -c "import eeg_comet; print(eeg_comet.__version__)"`)
   - Complete error message
   - Steps to reproduce the issue

---

## Updating EEG-COMET

To update to the latest version:

```bash
# Navigate to the repository directory
cd eeg-comet

# Pull latest changes
git pull origin stable

# Update dependencies
conda activate eeg-comet
pip install -r requirements.txt --upgrade
```

---

## Next Steps

After successful installation:

1. [**Getting Started Guide**]({% link getting-started.md %}) - Run your first microstate analysis
2. [**Module Documentation**]({% link modules/index.md %}) - Learn about each processing module
3. [**Parameters Reference**]({% link parameters.md %}) - Explore all configuration options

