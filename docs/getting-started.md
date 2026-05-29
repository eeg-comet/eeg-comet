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
| **File Extension** | `.set`, `.fif`, `.edf`, or `.auto` for automatic detection |
| **Data Type** | `raw` for continuous or `epoched` for segmented data |

3. Click **"Create Study"** to initialize

{: .note }
> EEG-COMET recursively searches the input folder, so you can organize data in subfolders by subject or condition.

---

## Step 3: Load and Preprocess Data

### Loading Data

- Click **"Load Data"** to import all detected EEG files
- The status panel shows the number of files found and loaded
- Channel configurations are automatically verified across datasets

### Preprocessing (Optional)

If additional filtering is needed:

1. **Temporal Filtering:**
   - Select filter type: `FIR` (recommended) or `IIR`
   - Set frequency range (e.g., 2-20 Hz for microstate analysis)

2. **Spatial Filtering:**
   - Enable k-nearest-neighbor smoothing if data is noisy
   - Set smoothing distance (typically 3-5 neighbors)

{: .highlight }
> Zero-phase filtering is automatically applied to preserve temporal accuracy of microstate transitions.

---

## Step 4: Select Clustering Data

Choose which timepoints to use for microstate template extraction:

| Method | Best For | Description |
|:-------|:---------|:------------|
| **GFP Peaks** | Traditional analysis, replication studies | Uses only high-amplitude moments |
| **Random Sample** | Balanced efficiency/coverage | Randomly samples timepoints for each iteration |
| **All Timepoints** | Maximum coverage, event-related designs | Uses complete temporal information |

Configure the **percentage of data** to use (50% default) for computational efficiency.

---

## Step 5: Determine Optimal Cluster Number

### Automatic Validation

1. Set the **cluster range** (e.g., K = 4 to 8)
2. Click **"Run Validation"**
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
| **Automatic** | Use a specific criterion (e.g., Cross-Validation) |
| **Majority Vote** | Select the K most frequently chosen across all criteria |

{: .note }
> Evidence suggests 4 or fewer microstates may oversimplify dynamics. Consider K ≥ 5 for more reliable results.

---

## Step 6: Extract Microstate Templates

### Configure Clustering

| Parameter | Recommended | Description |
|:----------|:------------|:------------|
| **Algorithm** | Modified K-means | Fast iterative refinement |
| **Number of Repeats** | 10-50 | Multiple runs to avoid local minima |
| **Max Iterations** | 500 | Convergence limit per run |
| **Tolerance** | 1e-6 | Convergence threshold |

### Alternative: TAAHC

For more thorough solution-space exploration, use **TAAHC** (Topographic Atomize and Agglomerate Hierarchical Clustering):
- More computationally intensive
- Better at escaping local minima
- Recommended when K-means shows instability

Click **"Run Clustering"** to extract templates.

---

## Step 7: Label Microstate Templates

### Automated Classification (Recommended)

1. Click **"Auto-Label"**
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

1. Click **"Run Backfitting"**
2. View the initial segmentation in the time-series viewer

### Segment Refinement

Configure minimum segment duration to remove implausibly short segments:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| **Min Duration** | 20 ms | Segments shorter than this are refined |
| **Strategy** | Smooth | How short segments are handled |

**Strategy Options:**

| Strategy | Use Case |
|:---------|:---------|
| **Remove** | Resting-state (gaps acceptable) |
| **Replace High** | Extend neighboring microstate with higher occurrence |
| **Replace Half** | Split between neighbors |
| **Smooth** | Temporal smoothing considering context |

---

## Step 9: Extract Features

### Standard Features

Select which metrics to compute:

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
| **Static** | Single value per recording |
| **Windowed** | Track changes over time windows |
| **Event-Related** | Trial-level analysis around events |

Click **"Extract Features"** to compute and export.

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

After processing, find in your output folder:

```
output_folder/
├── study_name/
│   ├── microstates/           # Template topographies
│   ├── segmentation/          # Backfitting results
│   ├── features/              # Extracted metrics
│   ├── statistics/            # Statistical analysis results
│   └── logs/                  # Processing logs for reproducibility
```

---

## Example Workflow Summary

```
1. Create Study → Define input/output paths
2. Load Data → Import preprocessed EEG files
3. Preprocess → Apply bandpass filter (2-20 Hz)
4. Select Data → Use GFP peaks for traditional analysis
5. Validate K → Run 10 criteria, use majority vote
6. Cluster → Modified K-means, 20 repeats
7. Label → Auto-classify with CNN
8. Backfit → Assign templates, smooth short segments
9. Features → Extract COV, OCC, DUR, TP
10. Export → Save as CSV for statistical analysis
```

---

## Next Steps

Now that you've completed your first analysis:

- **[Module Documentation]({% link modules/index.md %})** - Deep dive into each module
- **[Parameters Reference]({% link parameters.md %})** - Explore all configuration options
- **[Statistical Analysis]({% link modules/statistical-analysis.md %})** - Learn about inference methods
- **[Source Localization]({% link modules/source-localization.md %})** - Map microstates to cortical sources

