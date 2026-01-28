# ☄️EEG-COMET☄️
# EEG Comprehensive Microstate Extraction Toolbox

**Organization:** SFU eBrain Lab ([www.ebrainlab.ca](https://www.ebrainlab.ca))

---

## Overview
EEG-COMET is a comprehensive, open-source toolbox that addresses fundamental methodological challenges in EEG microstate analysis. Unlike existing tools that provide only basic functionality, EEG-COMET integrates all essential analytical steps into a unified framework: quality control, preprocessing, objective microstate labeling, advanced feature extraction, and source localization. Developed by the SFU eBrain Lab, it eliminates subjective manual processes through machine learning algorithms and provides three distinct analytical approaches to capture brain network dynamics previously obscured by methodological limitations, significantly improving reproducibility across microstate research.

## Features
- **Comprehensive analytical framework** - Integrates quality control, preprocessing, automated assignment, transient segment management, and feature extraction
- **Machine learning-based microstate labeling** - Objective assignment to canonical microstate classes eliminating subjective visual inspection
- **Advanced temporal analysis**:
  - **Traditional full-recording analysis** - Standard microstate extraction across entire EEG sessions
  - **Window-based dynamic tracking** - Segments recordings into temporal windows to capture microstate feature changes through time and different conditions
  - **Single-trial event-related analysis** - Extracts microstates from individual trials to preserve trial-to-trial variability and enable millisecond-precision tracking
- **Sophisticated preprocessing** - Built-in quality assessment and flexible data selection
- **Statistical optimization** - Automated determination of optimal microstate numbers
- **Comprehensive feature extraction** - Including novel complexity measures and sequence dynamics
- **Neural source localization** - Integrated source identification tools
- **Intuitive GUI** - Accessible to researchers regardless of programming expertise
- **Reproducibility tools** - Automated processing logs and empirically validated parameters
- **Cross-platform support** - Windows, macOS, and Linux compatibility


## Requirements
- Python 3.10+
- See `requirements.txt` for full dependencies

## Installation Steps

- #### Clone or download the repository
- #### Navigate to the project directory

### Using Conda (Recommended)
#### Prerequisites
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Create and activate the conda environment
```sh
conda env create -f environment.yml
conda activate eegcomet
```

### Using pip
- Create and activate the Python's built-in virtual environment

#### Windows
```sh
python -m venv eegcomet
eegcomet\Scripts\activate
pip install -r requirements.txt
```

#### macOS/Linux
```sh
python3 -m venv eegcomet
source eegcomet/bin/activate
pip install -r requirements.txt
```

## Usage
### Launch the Toolbox GUI
```sh
# Navigate to the EEG_COMET directory
cd EEG_COMET

# Launch the graphical interface
python main.py
```

## Support
For questions, bug reports, or contributions, please contact the authors or visit [www.ebrainlab.ca](https://www.ebrainlab.ca).
