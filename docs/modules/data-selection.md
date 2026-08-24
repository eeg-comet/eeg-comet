---
title: 3. Data Selection
layout: default
parent: Modules
nav_order: 3
description: "Clustering Data Selection Module - Timepoint selection for clustering"
---

# Clustering Data Selection Module
{: .no_toc }

Three complementary methods for selecting timepoints for microstate template extraction.
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

Researchers can choose among three complementary methods to determine which timepoints contribute to microstate template extraction:

1. **GFP Peak Selection** - Traditional approach using high-amplitude moments
2. **Random Subsampling** - Balanced efficiency with topographic diversity
3. **All Timepoints** - Complete temporal coverage

Each method is optimized for different experimental setups and analytical goals.

---

## Method 1: GFP Peak Selection

### Concept

Global Field Power (GFP) measures the spatial standard deviation of scalp potentials at each time point (Lehmann & Skrandies, 1980; Skrandies, 1990). GFP peaks represent moments of maximum field strength, offering optimal signal-to-noise ratio for identifying topographic patterns, and have been the traditional basis for microstate template extraction since the earliest segmentation work (Lehmann et al., 1987; Pascual-Marqui et al., 1995).

### Algorithm

1. Calculate GFP at each timepoint: $$GFP(t) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(V_i(t) - \bar{V}(t))^2}$$
2. Identify local maxima in GFP time series
3. Optionally enforce a minimum inter-peak interval
4. Use only peak timepoints for clustering

### Advantages

| Benefit | Description |
|:--------|:------------|
| **High SNR** | Maximum field strength moments |
| **Traditional** | Aligns with established literature |
| **Computational efficiency** | Reduced data volume |
| **Cross-study comparability** | Standard approach in field |

### Limitations

| Limitation | Impact |
|:-----------|:-------|
| **Data exclusion** | Inter-peak information lost |
| **Artifact vulnerability** | High-amplitude artifacts may be included |
| **Constrained exploration** | Same peaks used in all iterations |
| **Potential bias** | May miss low-amplitude but distinct states |

### Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `smoothing_gfp` | Enforce a minimum spacing between detected peaks | `False` |
| `smoothing_distance` | Minimum spacing between peaks, in milliseconds | `10` |

### Best For

- Replication of traditional microstate studies
- Cross-study comparisons requiring methodological alignment
- Computationally constrained environments
- Exploratory analyses following established conventions

---

## Method 2: Random Subsampling

### Concept

Randomly select different samples from the entire temporal dataset for each independent clustering iteration. This creates topographical variability essential for robust exploration of the solution space.

### Algorithm

1. For each clustering iteration:
   - Randomly sample N timepoints from full dataset
   - Run clustering on this sample
2. Evaluate templates using global explained variance across **all** timepoints
3. Select best solution based on full-data evaluation

### Advantages

| Benefit | Description |
|:--------|:------------|
| **Solution diversity** | Different samples per iteration |
| **Decorrelated attempts** | Avoids local minima traps |
| **Balanced efficiency** | Computational savings with coverage |
| **Full evaluation** | Templates scored on complete data |

### Implementation Details

- Each iteration uses a fresh random sample
- Global explained variance calculated on full dataset
- Particularly effective for resting-state protocols
- User-adjustable sample size (percentage of total timepoints)

### Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `data_percentage` | Percentage of data to sample | `100` |
| Sample size | Computed from percentage | Varies |

### Best For

- Resting-state analyses with distributed activity
- Studies prioritizing robust exploration
- Balancing computational cost with coverage
- When GFP peaks may not capture all states

---

## Method 3: All Timepoints

### Concept

Analyze every data sample during clustering, eliminating sampling bias and ensuring complete representation of the neural state space.

### Advantages

| Benefit | Description |
|:--------|:------------|
| **No sampling bias** | Every moment equally considered |
| **Complete coverage** | Inter-peak patterns included |
| **Precise temporal dynamics** | Essential for event-related designs |
| **No assumptions** | Avoids arbitrary peak selection |

### Considerations

| Factor | Note |
|:-------|:-----|
| **Computational cost** | Higher than sampling methods |
| **Memory requirements** | Full dataset must be processed |
| **Best for shorter sessions** | Or when resources permit |

### Best For

- Event-related designs requiring temporal precision
- Post-stimulus analyses where peaks may miss important patterns
- Shorter recording sessions
- When computational resources are available
- Studies where sampling assumptions are problematic

---

## Method Comparison

| Aspect | GFP Peaks | Random Sample | All Timepoints |
|:-------|:----------|:--------------|:---------------|
| **Data usage** | ~5-10% | User-defined | 100% |
| **Computation** | Low | Medium | High |
| **Bias** | Peak-biased | Minimal | None |
| **Variability** | Same each run | Different each run | Same each run |
| **Inter-peak states** | Excluded | Included (partially) | Included |
| **Traditional** | Yes | Newer | Newer |

---

## Configuration

### Using Config File

```ini
[clustering_config]
# Percentage of data to use
# 100 = all timepoints; lower values draw a random subset
data_percentage = 100

# GFP peak spacing (only applied when smoothing_gfp = True)
smoothing_gfp = False
smoothing_distance = 10
```

{: .note }
> GFP peak selection is chosen in the GUI, which overrides `data_percentage` for that run. `smoothing_gfp` and `smoothing_distance` affect only the GFP peak method.

When `n_maps = auto`, the cluster-number search always uses GFP peaks regardless of `data_percentage`.

### Practical Guidelines

| Recording Duration | Recommended Method | Rationale |
|:-------------------|:-------------------|:----------|
| < 2 minutes | All Timepoints | Sufficient computation, need all data |
| 2-10 minutes | Random Sample (50%) | Balance efficiency and coverage |
| > 10 minutes | GFP Peaks or Random | Computational considerations |
| Event-related | All Timepoints | Preserve temporal structure |

---

## GFP Peak Detection Details

### Peak Identification Algorithm

```
1. Compute GFP time series
2. Find all local maxima (points higher than neighbors)
3. Apply minimum interval constraint (when smoothing_gfp = True)
4. Return peak indices for clustering
```

Detection uses `scipy.signal.find_peaks` on the GFP time series. The only criterion applied is the optional minimum inter-peak distance; no prominence or height threshold is used, so every local maximum qualifies unless it is too close to an earlier peak.

### Minimum Interval

Prevents detection of multiple peaks within physiologically implausible intervals:

| Setting | Effect |
|:--------|:-------|
| `smoothing_gfp = False` *(default)* | No spacing constraint; every local maximum is retained |
| `smoothing_gfp = True` | Peaks must be at least `smoothing_distance` milliseconds apart |

`smoothing_distance` is specified in milliseconds and converted to samples using the sampling rate, so `smoothing_distance = 10` at 250 Hz enforces a spacing of 2 samples. Typical values are 10-20 ms.

---

## Impact on Clustering Quality

### Solution Space Exploration

| Method | Exploration Behavior |
|:-------|:--------------------|
| **GFP Peaks** | Same solution space each iteration |
| **Random Sample** | Different solution space each iteration |
| **All Timepoints** | Complete solution space, single configuration |

### Recommendations by Research Goal

| Goal | Recommended Method |
|:-----|:-------------------|
| Replicate published study | Match original method (usually GFP peaks) |
| Novel exploratory analysis | Random sampling for robustness |
| Event-related dynamics | All timepoints |
| Large dataset | GFP peaks or random (computational) |
| Template stability | Multiple runs with random sampling |

---

## Validation

Regardless of selection method, always validate clustering quality:

1. **Global Explained Variance** - Computed on ALL timepoints
2. **Template stability** - Across multiple runs
3. **Visual inspection** - Compare to canonical topographies

{: .note }
> Even when using GFP peaks for clustering, template quality is evaluated using the complete dataset to ensure microstates generalize beyond peak moments.

---

## Best Practices

1. **Match method to paradigm**
   - Event-related → All timepoints
   - Resting-state → GFP peaks or random

2. **Document your choice**
   - Critical for reproducibility
   - May affect comparability with other studies

3. **Consider multiple approaches**
   - Run with different methods
   - Compare template stability

4. **Validate thoroughly**
   - Use full-dataset GEV regardless of method
   - Check template interpretability

---

## References

- Lehmann, D., & Skrandies, W. (1980). Reference-free identification of components of checkerboard-evoked multichannel potential fields. *Electroencephalography and Clinical Neurophysiology*, 48(6), 609–621. [https://doi.org/10.1016/0013-4694(80)90419-8](https://doi.org/10.1016/0013-4694(80)90419-8)
- Skrandies, W. (1990). Global field power and topographic similarity. *Brain Topography*, 3(1), 137–141. [https://doi.org/10.1007/BF01128870](https://doi.org/10.1007/BF01128870)
- Lehmann, D., Ozaki, H., & Pal, I. (1987). EEG alpha map series: Brain micro-states by space-oriented adaptive segmentation. *Electroencephalography and Clinical Neurophysiology*, 67(3), 271–288. [https://doi.org/10.1016/0013-4694(87)90025-3](https://doi.org/10.1016/0013-4694(87)90025-3)
- Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of brain electrical activity into microstates: model estimation and validation. *IEEE Transactions on Biomedical Engineering*, 42(7), 658–665. [https://doi.org/10.1109/10.391164](https://doi.org/10.1109/10.391164)
- Murray, M. M., Brunet, D., & Michel, C. M. (2008). Topographic ERP analyses: A step-by-step tutorial review. *Brain Topography*, 20(4), 249–264. [https://doi.org/10.1007/s10548-008-0054-5](https://doi.org/10.1007/s10548-008-0054-5)
- Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: A review. *NeuroImage*, 180, 577–593. [https://doi.org/10.1016/j.neuroimage.2017.11.062](https://doi.org/10.1016/j.neuroimage.2017.11.062)

---

## Next Step

[**Cluster Number Validation Module →**]({% link modules/cluster-validation.md %})

