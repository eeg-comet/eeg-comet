---
title: 1. Data Loading
layout: default
parent: Modules
nav_order: 1
description: "Data Loading Module - Automated data identification and import"
---

# Data Loading Module
{: .no_toc }

Automated data identification and integration across multiple input formats.
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

## Overview

The data loading module provides automated data identification and integration across multiple input formats, supporting various data management practices. It eliminates manual file conversion and enables direct integration with existing preprocessing workflows.

{: .warning }
> **Critical Prerequisite:** Researchers must ensure proper artifact rejection before loading data, as artifact contamination can severely compromise all downstream microstate analyses.

---

## Artifact Rejection Prerequisites

Given the diverse experimental paradigms supported by EEG-COMET, proper preprocessing is essential before data import.

### Why Artifact Rejection Matters

| Artifact Type | Impact on Microstates |
|:--------------|:----------------------|
| **Eye blinks** | Create characteristic frontal patterns incorrectly classified as distinct microstates |
| **Horizontal eye movements** | Generate artificial lateral asymmetries interpreted as separate classes |
| **Muscle activity** | High-frequency contamination distorts topographic patterns |
| **Cardiac artifacts** | Rhythmic contamination affects temporal dynamics |

### Recommended Preprocessing

Before loading data into EEG-COMET:

1. ✅ Apply Independent Component Analysis (ICA)
2. ✅ Remove ocular artifact components
3. ✅ Remove muscle/cardiac artifact components
4. ✅ Perform careful visual inspection of cleaned data
5. ✅ Reject epochs/segments with residual artifacts

{: .important }
> Artifact-contaminated templates spread systematic errors throughout all downstream analyses of microstate dynamics.

---

## Supported Data Formats

EEG-COMET supports data formats compatible with MNE-Python and popular neuroimaging platforms:

### Native Formats

| Format | Extension | Platform |
|:-------|:----------|:---------|
| EEGLAB | `.set` (with `.fdt`) | EEGLAB/MATLAB |
| European Data Format | `.edf` | General |
| BioSemi Data Format | `.bdf` | BioSemi |
| General Data Format | `.gdf` | General |
| BrainVision | `.vhdr`, `.vmrk`, `.eeg` | BrainVision Analyzer |
| Neuroscan | `.cnt` | Neuroscan |
| EGI simple binary | `.egi` | Electrical Geodesics |
| EGI MFF | `.mff` | Electrical Geodesics |
| Nicolet | `.data` | Nicolet |
| eXimia | `.nxe` | Nexstim |
| Persyst | `.lay` (with `.dat`) | Persyst |

{: .note }
> Epoched datasets (`data_type = epoched`) are read with `mne.io.read_epochs_eeglab`, so EEGLAB `.set` is the only supported format for epoched input. All formats above are available for continuous (`raw`) data.

### Platform Integration

| Platform | Integration |
|:---------|:------------|
| **Brainstorm** | Exported `.mat` channel files used as a channel-location source |
| **BIDS** | Complete BIDS-formatted datasets |

### BIDS Support

EEG-COMET provides full support for [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) formatted datasets:

- Automated metadata extraction
- Consistent organization across subjects
- Enhanced reproducibility
- Integration with standardized workflows

---

## Configuration Parameters

### Study Settings

| Parameter | Description | Options |
|:----------|:------------|:--------|
| `study_name` | Name of the analysis study | Any descriptive string |
| `input_folder` | Directory containing EEG files | Valid path |
| `output_folder` | Directory for saving results | Valid path |
| `channel_location_dir` | Path to channel location file | Path or empty for embedded |

### File Selection

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `extension` | File extension to search for | `.auto` |
| `load_all_files` | Load all matching files | `True` |
| `pattern_content` | Filename pattern filter | `*` (all files) |
| `data_type` | Data type (`raw` or `epoched`) | `raw` |

### Extension Options

| Value | Behavior |
|:------|:---------|
| `.auto` | Automatically detect any of `.vhdr`, `.edf`, `.bdf`, `.gdf`, `.cnt`, `.egi`, `.mff`, `.set`, `.data`, `.nxe`, `.lay` |
| `.set` | Load only EEGLAB files |
| `.edf` | Load only EDF files |
| `.vhdr` | Load only BrainVision files |

---

## File Discovery

### Recursive Directory Search

EEG-COMET recursively searches the input folder to locate EEG files:

```
input_folder/
├── subject_01/
│   ├── session_01/
│   │   └── eeg_data.set    ✓ Found
│   └── session_02/
│       └── eeg_data.set    ✓ Found
├── subject_02/
│   └── eeg_data.set        ✓ Found
└── derivatives/
    └── preprocessed.set    ✓ Found (if matches pattern)
```

### Pattern Filtering

Use `pattern_content` to selectively load specific files:

| Pattern | Matches |
|:--------|:--------|
| `*` | All files with specified extension |
| `*_rest_*` | Files containing "_rest_" |
| `sub-01*` | Files starting with "sub-01" |
| `*_clean*` | Files containing "_clean" |

**Example:** To load only resting-state data:
```ini
pattern_content = *_rest_*
```

---

## Electrode Configuration

### Automatic Verification

EEG-COMET automatically verifies electrode configurations across all loaded datasets:

- Channel names must match across subjects
- Channel count must be consistent
- Spatial locations are validated

### Handling Mismatches

If electrode configurations differ:

| Issue | Solution |
|:------|:---------|
| Missing channels | Remove from all datasets or interpolate |
| Extra channels | Exclude non-EEG channels |
| Different names | Standardize naming convention |

### Channel Location Files

If locations are not embedded in data files, provide a separate file:

**Supported formats:**
- `.csv`, `.tsv`, `.txt`, `.xyz` - Delimited coordinate tables
- `.sfp`, `.elp` - BESA / Polhemus formats
- `.elc` - ASA electrode file
- `.loc`, `.locs`, `.eloc`, `.ced` - EEGLAB channel-location files
- `.csd`, `.bvef` - CSD and BrainVision electrode files
- `.mat` - Brainstorm channel structure

**CSV format example:**
```csv
label,x,y,z
Fp1,-0.0295,0.0832,0.0020
Fp2,0.0295,0.0832,0.0020
F7,-0.0593,0.0528,-0.0233
...
```

---

## Usage Examples

### Basic Loading

Load all EEGLAB files from a directory:

```ini
[io_config]
study_name = my_study
input_folder = /path/to/data
output_folder = /path/to/output
extension = .set
pattern_content = *
data_type = raw
```

### Selective Loading

Load only specific sessions:

```ini
[io_config]
study_name = resting_analysis
input_folder = /path/to/data
output_folder = /path/to/output
extension = .set
pattern_content = *session01*
data_type = raw
```

### BIDS Dataset

Load from BIDS-formatted directory:

```ini
[io_config]
study_name = bids_study
input_folder = /path/to/bids_dataset/derivatives/preprocessed
output_folder = /path/to/output
extension = .auto
pattern_content = *_eeg*
data_type = raw
```

---

## Quality Control

After loading, EEG-COMET performs automatic quality checks:

| Check | Description |
|:------|:------------|
| File integrity | Verify all files can be read |
| Sampling rate | Ensure consistent sampling across files |
| Channel count | Verify matching electrode configurations |
| Data continuity | Check for gaps in continuous data |

### Viewing Loaded Data

Use the built-in inspection tools in the New Study window:

1. Select a file from the loaded list
2. Click **"Plot Data"** for the raw time series, **"Show Montage"** for the electrode layout, or **"Show Power Spectral Density"** for the PSD
3. Scroll through the time series and topographies
4. Verify data quality before proceeding

---

## Best Practices

1. **Organize data consistently** - Use clear folder structure and naming conventions
2. **Complete preprocessing first** - Artifact rejection must precede microstate analysis
3. **Verify electrode locations** - Accurate spatial information is critical for topographic analysis
4. **Use BIDS format** - Enhances reproducibility and metadata preservation
5. **Document preprocessing steps** - Keep records of all preprocessing applied before EEG-COMET

---

## Troubleshooting

<div class="callout warning">
<strong>Files not found</strong><br>
Check that the extension and pattern match your files. Try <code>extension = .auto</code> first.
</div>

<div class="callout warning">
<strong>Channel mismatch error</strong><br>
Ensure all subjects have the same electrode montage. Remove or interpolate missing channels before loading.
</div>

<div class="callout warning">
<strong>Memory error on load</strong><br>
For very large datasets, consider loading subjects in batches or downsampling during preprocessing.
</div>

---

## Next Step

[**Data Preparation Module →**]({% link modules/data-preparation.md %})

