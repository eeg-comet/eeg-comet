---
title: Getting Started
layout: default
nav_order: 3
description: "Quick start tutorial for EEG-COMET"
---

# Getting Started
{: .no_toc }

A step-by-step tutorial to run your first microstate analysis with EEG-COMET.
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

## Prerequisites

Before starting, ensure you have:

1. ✅ **EEG-COMET installed** - See the [Installation Guide]({% link installation.md %})
2. ✅ **Preprocessed EEG data** - Artifact-rejected and cleaned
3. ✅ **Channel locations** - Either embedded in data or as a separate file

{: .warning }
> **Critical:** EEG-COMET assumes artifact rejection has been completed. Ocular artifacts, muscle activity, and other contaminants will severely compromise microstate analysis.

---

## Step 1: Launch EEG-COMET

Activate your environment and start the GUI. Both entry points open the same interface:

```bash
conda activate eeg-comet

# Option A — console script (after `pip install -e .` or `pip install eeg-comet`)
eeg-comet

# Option B — run the package module directly
python -m eeg_comet.main
```

The main window will appear with the processing pipeline interface.

---

## Step 2: Create a New Study

1. Click **"New Study"** in the main window
2. Configure your study settings:

| Setting | Description |
|:--------|:------------|
| **Study Name** | A descriptive name for your analysis |
| **Input Folder** | Directory containing your EEG files |
| **Output Folder** | Where results will be saved |
| **File Extension** | `.set`, `.edf`, `.vhdr`, `.bdf`, `.gdf`, `.cnt`, `.egi`, `.mff`, `.data`, `.nxe`, `.lay`, or `.auto` for automatic detection |
| **Data Type** | **Continuous EEG** for unsegmented recordings, or **Epoched EEG** for data already segmented into trials (EEGLAB `.set` only) |

3. Click **"Find Data"** to scan the input folder and build the file list

{: .note }
> EEG-COMET recursively searches the input folder, so you can organize data in subfolders by subject or condition.

---

## Step 3: Load and Preprocess Data

### Loading Data

- **"Find Data"** imports every detected EEG file from the input folder and its subfolders
- The import log shows the number of files found and loaded
- Channel configurations are automatically verified across datasets

### Preprocessing (Optional)

If additional filtering is needed:

1. **Temporal Filter (Band-Pass):**
   - Select filter type: `FIR` (recommended) or `IIR`
   - Set the lowcut/highcut frequencies (e.g., 2-20 Hz for microstate analysis)

2. **Spatial Filter:**
   - Enable **"Spatial Filter"** to average each electrode with its nearest neighbours if data is noisy

Click **"Preprocess All Data"** to apply the selected steps.

{: .highlight }
> Zero-phase filtering is automatically applied to preserve temporal accuracy of microstate transitions.

---

## Step 4: Select Clustering Data

Choose which timepoints to use for microstate template extraction:

| Method | Best For | Description |
|:-------|:---------|:------------|
| **Use GFP Peaks Only** | Traditional analysis, replication studies | Uses only high-amplitude moments. **Min Peak Distance (ms)** sets the minimum spacing between retained GFP peaks (`smoothing_distance`, 10 ms default) |
| **Use Random Subset of Data** | Balanced efficiency/coverage | Randomly samples timepoints for each iteration, controlled by **Subset Fraction (%)** |
| **All timepoints** | Maximum coverage, event-related designs | Select **Use Random Subset of Data** and set the fraction to 100% |

The subset fraction maps to `data_percentage`, which defaults to 100 (the whole recording). Lower it to trade coverage for computational efficiency.

{: .note }
> By default GFP-peak restriction is off (`smoothing_gfp = False`) and `data_percentage = 100`, so clustering uses every timepoint unless you change these settings.

---

## Step 5: Determine Optimal Cluster Number

### Automatic Validation

1. Click **"Inspect Criteria for Optimal Microstates"**
2. Confirm the **cluster range** (e.g., K = 4 to 8) in the prompt
3. EEG-COMET evaluates 10 statistical criteria:
   - Cross-Validation Criterion (CV)
   - Global Explained Variance (GEV)
   - Silhouette Score
   - Dunn Index
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
   - Gap Statistic
   - AIC/BIC
   - Krzanowski-Lai Criterion

### Selection Strategies

| Strategy | Description |
|:---------|:------------|
| **Manual Inspection** | View validation curves and decide based on domain knowledge |
| **Single Method** | Use one criterion (e.g., Cross-Validation) as the **Stop Condition** |
| **Ensemble Method** | Select the K most frequently chosen across all criteria (majority vote) |

To let EEG-COMET pick K during clustering, choose **"Auto-detect"**, set the k range and **Stop Condition**, then run clustering.

{: .note }
> Evidence suggests 4 or fewer microstates may oversimplify dynamics. Consider K ≥ 5 for more reliable results.

---

## Step 6: Extract Microstate Templates

### Configure Clustering

| Parameter | Recommended | Description |
|:----------|:------------|:------------|
| **Algorithm** | Modified K-Means Clustering | Fast iterative refinement |
| **Number of Repeats** | 10-50 | Multiple runs to avoid local minima |
| **Max Iterations** | 500 | Convergence limit per run |
| **Tolerance** | 1e-6 | Convergence threshold |

Three algorithms are available: **Modified K-Means Clustering**, **Modified K-Means Clustering with Spatial Similarity** (adds a configurable **Similarity Metric** — Spatial Correlation or Cosine Similarity), and **Topographic Atomize and Agglomerate Hierarchical Clustering**.

### Alternative: TAAHC

For more thorough solution-space exploration, use **TAAHC** (Topographic Atomize and Agglomerate Hierarchical Clustering):
- More computationally intensive
- Better at escaping local minima
- Recommended when K-means shows instability

Click **"Start Clustering"** to extract templates.

---

## Step 7: Label Microstate Templates

### Automated Classification (Recommended)

1. Open **"View Microstate Maps"** and click **"Automatically Label the Microstates"**
2. The CNN classifier (bundled as `eeg_comet/models/model_v2.onnx`, run via ONNX Runtime) assigns canonical labels (A, B, C, D, E, F, G)
3. Review assignments in the visualization panel

The classifier achieves **98%+ accuracy** on validation datasets aggregated from 1,157 subjects.

### Manual Labeling

For atypical topographies or exploratory analyses:
1. Click on each template
2. Assign labels manually based on visual inspection
3. Compare with canonical topographies shown in the reference panel

---

## Step 8: Backfitting and Segmentation

### Template Assignment

Each timepoint is assigned to the template with highest spatial correlation:

1. Click **"Start Backfitting"**
2. View the initial segmentation in the time-series viewer

### Segment Refinement

Enable **"Filter Transient Segments"** (off by default) to remove implausibly short segments:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| **Reject Segments < (ms)** | 20 ms | Segments shorter than this are refined |
| **Filter Segments Method** | Smooth segments | How short segments are handled |

**Strategy Options:**

| Strategy | Use Case |
|:---------|:---------|
| **Remove short segments** | Resting-state (gaps acceptable) |
| **Replace short segments: nearby dominant microstate** | Extend neighboring microstate with higher occurrence |
| **Replace short segments: half and half** | Split between neighbors |
| **Smooth segments** | Temporal smoothing considering context |

---

## Step 9: Extract Features

### Standard Features

Select which metrics to compute. **OCC**, **DUR**, and **COV** are selected by default:

| Feature | Abbreviation | Description |
|:--------|:-------------|:------------|
| Coverage | COV | Proportion of time in each microstate |
| Occurrence | OCC | Number of appearances per second |
| Duration | DUR | Average length of each microstate (default uses geometric mean of run lengths; see [Feature Extraction]({% link modules/feature-extraction.md %}#mean-duration-dur) for `arithmetic`, `median`, `trimmed_mean` alternatives) |
| GEV | GEV | Variance explained by each template |
| Transition Probability | TP | Probabilities of microstate-to-microstate transitions |

### Advanced Metrics

| Feature | Abbreviation | Description |
|:--------|:-------------|:------------|
| Entropy Rate | ER | Predictability of microstate sequences |
| Lempel-Ziv Complexity | LZC | Sequence diversity measure |
| Hurst Exponent | HE | Long-range temporal correlations |

### Analysis Modes

| Mode | Use Case |
|:-----|:---------|
| **Extract Global Features (One per File)** | Single value per recording |
| **Extract Windowed Features (Multiple per File)** | Track changes over time windows |
| **Extract Pre/Post Event Features** | Trial-level analysis around events |

By default only real (observed) features are computed; tick **"Extract Features from Synthetic Data as Well"** to add surrogate and random sequences for statistical comparison.

Click **"Extract Selected Features"** to compute and export.

---

## Step 10: Export Results

### Output Formats

EEG-COMET exports results in multiple formats:

| Format | Use Case |
|:-------|:---------|
| `.csv` | Excel, R, general analysis |
| `.pkl` | Python (pandas DataFrame) |
| `.hdf` | Large datasets, hierarchical storage |
| `.json` | Web applications, APIs |

### Output Files

After processing, find in your output folder. Every per-stage folder is prefixed with your study name:

```
output_folder/
├── study_name/
│   ├── study_name_preprocessed_data/    # Filtered, resampled, re-referenced recordings
│   ├── study_name_clustering_results/   # Template topographies (microstate_maps.csv)
│   ├── study_name_segmentation/         # Backfitting results
│   ├── study_name_extracted_features/   # Extracted metrics
│   ├── study_name_localized_sources/    # Source estimates
│   │   ├── tess_sources/
│   │   └── avg_sources/
│   ├── eeg_info.fif                     # Channel/montage information for the study
│   ├── eeg_comet_config.ini             # Settings used for this study
│   └── eeg_comet_log.txt                # Processing log for reproducibility
```

---

## Example Workflow Summary

```
1. New Study → Define input/output paths
2. Find Data → Import preprocessed EEG files
3. Preprocess All Data → Apply bandpass filter (2-20 Hz)
4. Select Data → Use GFP peaks for traditional analysis
5. Validate K → Run 10 criteria, use majority vote
6. Start Clustering → Modified K-Means Clustering, 20 repeats
7. Label → Auto-classify with CNN
8. Start Backfitting → Assign templates, smooth short segments
9. Extract Selected Features → COV, OCC, DUR, TP
10. Export → Save as CSV for statistical analysis
```

---

## Next Steps

Now that you've completed your first analysis:

- **[Module Documentation]({% link modules/index.md %})** - Deep dive into each module
- **[Parameters Reference]({% link parameters.md %})** - Explore all configuration options
- **[Statistical Analysis]({% link modules/statistical-analysis.md %})** - Learn about inference methods
- **[Source Localization]({% link modules/source-localization.md %})** - Map microstates to cortical sources

