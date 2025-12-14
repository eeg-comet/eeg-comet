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
| **Python** | 3.10 or higher |
| **RAM** | 8 GB minimum (16 GB recommended) |
| **Storage** | 5 GB for installation + space for data |
| **Display** | 1920×1080 resolution recommended |

### Recommended Specifications

For optimal performance, especially with large datasets or source localization:

- **RAM:** 16-32 GB
- **CPU:** Multi-core processor (4+ cores)
- **GPU:** Not required, but CUDA-compatible GPU accelerates TensorFlow operations

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
git clone https://github.com/eeg-comet/eeg-comet.github.io.git
cd eeg-comet.github.io
```

{: .note }
> Alternatively, download the repository as a ZIP file from GitHub and extract it.

**Step 2: Create the Conda Environment**

```bash
conda env create -f environment.yml
```

This creates an environment named `eegcomet` with all required dependencies.

**Step 3: Activate the Environment**

```bash
conda activate eegcomet
```

**Step 4: Verify Installation**

```bash
python -c "import mne; import sklearn; import tensorflow; print('Installation successful!')"
```

---

### Method 2: pip Installation

Use pip if you prefer Python's built-in virtual environment or don't have Conda installed.

#### Windows

```powershell
# Clone the repository
git clone https://github.com/eeg-comet/eeg-comet.github.io.git
cd eeg-comet.github.io

# Create virtual environment
python -m venv eegcomet

# Activate the environment
.\eegcomet\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### macOS / Linux

```bash
# Clone the repository
git clone https://github.com/eeg-comet/eeg-comet.github.io.git
cd eeg-comet.github.io

# Create virtual environment
python3 -m venv eegcomet

# Activate the environment
source eegcomet/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Launching EEG-COMET

After installation, launch the graphical interface:

```bash
# Navigate to the EEG_COMET directory
cd EEG_COMET

# Launch the GUI
python main.py
```

The main window should appear:

{: .highlight }
> On first launch, EEG-COMET may take a few moments to load as it initializes the neural network model for microstate classification.

---

## Key Dependencies

EEG-COMET relies on several major scientific Python packages:

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `mne` | 1.8.0 | EEG data handling and processing |
| `mne-bids` | 0.16.0 | BIDS format support |
| `scikit-learn` | 1.5.0 | Clustering algorithms and validation |
| `tensorflow` | 2.16.2 | Neural network for microstate classification |
| `onnxruntime` | 1.19.0 | Optimized model inference |
| `statsmodels` | 0.14.4 | Statistical analysis (GEE, LMM) |
| `pyvista` | 0.45.2 | 3D visualization for source localization |
| `PyQt5` | 5.15.11 | Graphical user interface |
| `pandas` | 2.3.0 | Data manipulation and export |
| `seaborn` | 0.13.2 | Statistical visualizations |

---

## Troubleshooting

### Common Issues

<div class="callout warning">
<strong>Qt Platform Plugin Error (Windows)</strong><br>
If you see "qt.qpa.plugin: Could not find the Qt platform plugin", try:
<pre>conda install -c conda-forge qt</pre>
</div>

<div class="callout warning">
<strong>TensorFlow GPU Not Detected</strong><br>
EEG-COMET works without GPU acceleration. If you want GPU support, install CUDA-compatible drivers and cuDNN. See the <a href="https://www.tensorflow.org/install/gpu">TensorFlow GPU guide</a>.
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
conda env remove -n eegcomet

# Recreate from scratch
conda env create -f environment.yml
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/eeg-comet/eeg-comet.github.io/issues) for similar problems
2. Create a new issue with:
   - Your operating system and version
   - Python version (`python --version`)
   - Complete error message
   - Steps to reproduce the issue

---

## Updating EEG-COMET

To update to the latest version:

```bash
# Navigate to the repository directory
cd eeg-comet.github.io

# Pull latest changes
git pull origin main

# Update dependencies
conda activate eegcomet
pip install -r requirements.txt --upgrade
```

---

## Next Steps

After successful installation:

1. [**Getting Started Guide**]({% link getting-started.md %}) - Run your first microstate analysis
2. [**Module Documentation**]({% link modules/index.md %}) - Learn about each processing module
3. [**Parameters Reference**]({% link parameters.md %}) - Explore all configuration options

