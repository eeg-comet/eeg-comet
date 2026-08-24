---
title: FAQ
layout: default
nav_order: 6
description: "Frequently asked questions about EEG-COMET"
---

# Frequently Asked Questions
{: .no_toc }

Common questions and answers about EEG-COMET.
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

## General Questions

### What is EEG-COMET?

EEG-COMET (EEG Comprehensive Microstate Extraction Toolbox) is an open-source Python toolbox for comprehensive EEG microstate analysis. It integrates data preprocessing, microstate extraction, automated labeling, feature extraction, statistical analysis, and source localization into a unified framework with a graphical user interface.

### Who develops EEG-COMET?

EEG-COMET is developed by the [SFU eBrain Lab](https://www.ebrainlab.ca) at Simon Fraser University.

### Is EEG-COMET free to use?

Yes. EEG-COMET is free and open-source software released under the **GNU General Public License v3.0**. You are free to use, study, modify, and redistribute it. Any redistributed or modified version (including a fork or a tool that incorporates EEG-COMET) must also be released under GPL-3.0 with the copyright and license notices intact. For closed-source or otherwise GPL-incompatible use, please contact the authors to discuss a separate commercial license.

### What operating systems are supported?

EEG-COMET supports:
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+)

---

## Data Requirements

### What EEG file formats are supported?

EEG-COMET supports all major EEG formats through MNE-Python:
- EEGLAB (.set)
- European Data Format (.edf)
- BioSemi (.bdf) and General Data Format (.gdf)
- BrainVision (.vhdr)
- Neuroscan (.cnt)
- EGI (.egi, .mff)
- Nicolet (.data), eXimia (.nxe), Persyst (.lay)
- BIDS-formatted datasets

Epoched data is read with `mne.io.read_epochs_eeglab`, so EEGLAB `.set` is the only supported format for already-segmented input. The formats above all apply to continuous recordings.

### Do I need to preprocess my data before using EEG-COMET?

**Yes, artifact rejection is essential.** EEG-COMET includes temporal and spatial filtering, but you must complete artifact rejection (especially ICA-based removal of ocular artifacts) before loading data. Artifact-contaminated data will produce invalid microstate results.

### How much data do I need?

| Analysis Type | Minimum Recommended |
|:--------------|:--------------------|
| Resting-state | 2+ minutes per condition |
| Event-related | 30+ trials per condition |
| Cluster validation | 5+ subjects |

### What electrode density is needed?

- **Minimum:** 19 channels (10-20 system)
- **Recommended:** 64+ channels
- **Optimal:** 128+ channels for source localization

Note: Spatial filtering is not recommended for montages with fewer than 32 channels.

---

## Microstate Analysis

### How many microstates should I use?

The optimal number depends on your data and research question. EEG-COMET provides 10 validation criteria to help determine this. General guidance:

- **4 microstates:** Traditional, widely used
- **5-7 microstates:** More detailed network characterization
- **>7 microstates:** May capture additional variability but harder to interpret

Use the majority vote selection or examine validation curves to decide.

### What's the difference between K-means and TAAHC?

| Aspect | Modified K-means | TAAHC |
|:-------|:-----------------|:------|
| Speed | Fast | Slower |
| Initialization | Sensitive to random start | Deterministic |
| Local minima | May get trapped | Better at escaping |
| Recommendation | Use with multiple runs | Use for final solution |

### Why are my templates different across runs?

Clustering algorithms like K-means are sensitive to initialization. This is expected behavior. Solutions:

1. Increase `n_repeats` (20+)
2. Use K-Means++ initialization
3. Try TAAHC for more stable results
4. Verify data quality and preprocessing

### What does "polarity invariance" mean?

EEG reference schemes are arbitrary, meaning the same brain state can appear with opposite voltage polarity (positive vs. negative). Microstate analysis treats these as identical configurations by using absolute spatial correlation values.

---

## Automated Classification

### How accurate is the automated microstate classifier?

The CNN-based classifier achieves **>98% accuracy** on validation datasets from 1,157 subjects across multiple independent studies.

### What if my topography doesn't match a canonical class?

The classifier will assign the best-matching label with a lower confidence score. Options:

1. Check for artifacts or data quality issues
2. Use manual labeling for atypical patterns
3. Consider if your population/paradigm produces non-canonical states

### Can I classify more than 7 microstates?

The automated classifier supports up to 7 canonical classes (A-G). For more than 7, use manual labeling with your own nomenclature.

---

## Feature Extraction

### What features should I extract?

| Research Goal | Recommended Features |
|:--------------|:--------------------|
| Basic characterization | COV, OCC, DUR, GEV |
| Sequence analysis | TP (transition probabilities) |
| Complexity analysis | LZC, ER, HE |
| Complete analysis | All features |

### What's the difference between averaged and sliding mode?

- **Averaged:** One value per entire recording (e.g., average coverage)
- **Sliding:** Values computed in sliding windows, tracking changes over time

Use sliding mode for protocols with alternating conditions within a single recording.

### How is mean microstate Duration (DUR) computed?

By default, EEG-COMET reports the **geometric mean** of the per-segment run lengths (`duration_method = geometric`). Run-length distributions of dominant microstates are heavy-tailed, and the geometric mean tracks the typical persistence far better than the arithmetic mean.

Three other aggregations are available via `duration_method` in `[features_config]`:

| Method | Behaviour |
|:-------|:----------|
| `geometric` *(default)* | Geometric mean of run lengths × `1000/fs`. Robust to long-tail outliers. |
| `arithmetic` | `(mean_samples − 1) × 1000/fs`. Keeps `COV ≈ DUR × OCC` exactly. Use this to reproduce earlier EEG-COMET results. |
| `median` | Median of run lengths × `1000/fs`. Most robust central-tendency estimator. |
| `trimmed_mean` | 10% trimmed mean (falls back to mean for ≤10 segments). |

See [Feature Extraction → Mean Duration (DUR)]({% link modules/feature-extraction.md %}#mean-duration-dur) for the formulas and trade-offs.

### How do I analyze event-related microstate changes?

1. Use epoched data with event markers
2. Select the `pre_post` feature mode
3. Define pre-event and post-event windows
4. Extract ROF (Relative Occurrence Frequency) for temporal profiles
5. Use cluster-based permutation testing for statistics

---

## Statistical Analysis

### How do I compare groups?

For recording-level features (averaged mode):
1. Extract features for all subjects
2. Use independent t-tests for between-group comparisons
3. Apply FDR or Bonferroni correction for multiple comparisons

### How do I handle trial-level data?

Trial-level data has hierarchical structure (trials within subjects). Use:
- **GEE:** For population-averaged effects. Choose the Gaussian/identity entry for roughly symmetric measures, or Gamma/log for strictly positive, right-skewed ones such as coverage.
- **LMM:** For subject-specific effects

These account for within-subject correlations that violate t-test assumptions.

### What is cluster-based permutation testing?

A method for testing effects across time that:
1. Groups adjacent timepoints showing consistent effects
2. Computes cluster statistics
3. Compares to null distribution from permuted data
4. Controls family-wise error across all timepoints

---

## Source Localization

### Do I need individual MRI for source localization?

No, EEG-COMET can use standardized template anatomy. However, individual MRI provides more accurate localization, especially for:
- Clinical applications
- Individual differences research
- High-precision studies

### Which inverse method should I use?

| Method | Best For |
|:-------|:---------|
| dSPM | General use, interpretable units |
| sLORETA | Good localization, some smoothing |
| eLORETA | Best localization, most smoothing |
| MNE | Basic analysis, raw current density |

For most applications, **dSPM** is recommended.

---

## Technical Issues

### EEG-COMET won't start

Common solutions:
1. Verify Python environment is activated
2. Check all dependencies are installed
3. Update PyQt5: `pip install --upgrade PyQt5`
4. On Windows, try installing Qt: `conda install -c conda-forge qt`

### Memory errors with large datasets

Options:
1. Reduce sampling rate (downsample to 250 Hz)
2. Process subjects in batches
3. Use GFP peaks instead of all timepoints
4. Increase system RAM

### Clustering is very slow

Speed tips:
1. Reduce `data_percentage` (e.g., 30% instead of 100%)
2. Lower `n_repeats`
3. Use K-means instead of TAAHC
4. Apply downsampling

### Results differ from other toolboxes

Minor differences are expected due to:
- Different clustering implementations
- Random initialization effects
- Preprocessing differences
- Polarity handling

For replication, match:
- Frequency band
- Data selection method (peaks vs. all)
- Clustering algorithm and parameters
- Segment handling strategy

---

## Best Practices

### How can I ensure reproducibility?

1. **Document preprocessing:** Record all steps before EEG-COMET
2. **Save configuration:** Keep config files with results
3. **Use processing logs:** EEG-COMET logs all operations
4. **Report parameters:** Include all settings in methods section
5. **Version control:** Note EEG-COMET and dependency versions

### What should I report in publications?

Include in your methods:
- EEG-COMET version
- Frequency band used
- Cluster number and selection method
- Clustering algorithm and number of runs
- Labeling method (auto or manual)
- Segment duration threshold and strategy
- Features extracted
- Statistical tests and corrections

---

## Getting Help

### Where can I report bugs?

Submit issues on GitHub: [github.com/eeg-comet/eeg-comet/issues](https://github.com/eeg-comet/eeg-comet/issues)

Include:
- Operating system and version
- Python version
- Complete error message
- Steps to reproduce

### How can I contribute?

We welcome contributions! See the GitHub repository for:
- Bug reports
- Feature requests
- Code contributions
- Documentation improvements

### Where can I learn more about microstate analysis?

Key references:
- Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: A review. *NeuroImage*, 180, 577–593. [https://doi.org/10.1016/j.neuroimage.2017.11.062](https://doi.org/10.1016/j.neuroimage.2017.11.062)
- Khanna, A., Pascual-Leone, A., Michel, C. M., & Farzan, F. (2015). Microstates in resting-state EEG: Current status and future directions. *Neuroscience & Biobehavioral Reviews*, 49, 105–113. [https://doi.org/10.1016/j.neubiorev.2014.12.010](https://doi.org/10.1016/j.neubiorev.2014.12.010)
- Lehmann, D., Ozaki, H., & Pal, I. (1987). EEG alpha map series: Brain micro-states by space-oriented adaptive segmentation. *Electroencephalography and Clinical Neurophysiology*, 67(3), 271–288. [https://doi.org/10.1016/0013-4694(87)90025-3](https://doi.org/10.1016/0013-4694(87)90025-3)
- Murray, M. M., Brunet, D., & Michel, C. M. (2008). Topographic ERP analyses: A step-by-step tutorial review. *Brain Topography*, 20(4), 249–264. [https://doi.org/10.1007/s10548-008-0054-5](https://doi.org/10.1007/s10548-008-0054-5)
- Tarailis, P., Koenig, T., Michel, C. M., & Griškova-Bulanova, I. (2024). The functional aspects of resting EEG microstates: A systematic review. *Brain Topography*, 37(2), 181–217. [https://doi.org/10.1007/s10548-023-00958-9](https://doi.org/10.1007/s10548-023-00958-9)
- Michel, C. M., Brechet, L., Schiller, B., et al. (2024). Current state of EEG/ERP microstate research. *Brain Topography*, 37, 169–180. [https://doi.org/10.1007/s10548-024-01037-3](https://doi.org/10.1007/s10548-024-01037-3)

