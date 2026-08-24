---
title: 2. Data Preparation
layout: default
parent: Modules
nav_order: 2
description: "Data Preparation Module - Temporal and spatial filtering"
---

# Data Preparation Module
{: .no_toc }

Temporal and spatial filtering optimized for microstate analysis.
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

The Data Preparation module provides two complementary filtering approaches optimized for microstate analysis:

1. **Temporal Filtering** - Zero-phase bandpass filtering
2. **Spatial Filtering** - Nearest-neighbor electrode averaging

Both methods maintain the temporal and spatial integrity of input data, enabling unbiased characterization of microstate dynamics.

---

## Temporal Filtering

### Why Zero-Phase Filtering?

A major methodological concern in microstate analysis is how filters affect temporal accuracy.

| Filter Type | Phase Behavior | Impact on Microstates |
|:------------|:---------------|:----------------------|
| **Causal (one-way)** | Systematic phase shifts | Distorts timing of transitions, creates false sequence patterns |
| **Zero-phase (two-way)** | No phase distortion | Preserves accurate transition timing |

{: .important }
> EEG-COMET uses zero-phase filtering exclusively to eliminate temporal distortions that could artificially alter microstate transitions.

### Filter Types

#### FIR Filters (Recommended)

**Finite Impulse Response** filters have inherently zero-phase properties due to symmetric coefficients:

| Advantages | Considerations |
|:-----------|:---------------|
| Linear phase response | Higher computational cost |
| Precise frequency control | Longer edge effects |
| Ideal for event-related designs | Requires more filter coefficients |

**Best for:** Event-related paradigms, short recordings, precise temporal analysis

#### IIR Filters

**Infinite Impulse Response** filters achieve zero-phase via forward-backward filtering:

| Advantages | Considerations |
|:-----------|:---------------|
| Computationally efficient | Potential numerical instability |
| Shorter edge effects | Less precise frequency control |
| Good for long recordings | Forward-backward application doubles processing |

**Best for:** Long continuous recordings (resting-state), computational efficiency

### Frequency Bands

| Band | Frequency Range | Use Case |
|:-----|:----------------|:---------|
| **Broad-spectrum** | 1-40 Hz | Preserve slow potentials and fast oscillations |
| **Standard microstate** | 2-20 Hz | Traditional analysis, reduces drift and muscle artifacts |
| **Narrow** | 4-15 Hz | Focus on dominant microstate frequencies |

{: .note }
> The 2-20 Hz range is most commonly used in microstate research, highlighting key topographic patterns while reducing noise (Michel & Koenig, 2018; Khanna et al., 2015).

### Configuration Parameters

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `temporal_filter_data` | Enable temporal filtering | `True` | `True`, `False` |
| `filter_method` | Filter type | `fir` | `fir`, `iir` |
| `lowcut_freq` | High-pass cutoff (Hz) | `2` | 0.1 - 10 |
| `highcut_freq` | Low-pass cutoff (Hz) | `20` | 10 - 100 |

### Frequency Constraints

Filtering runs before resampling so that the low-pass doubles as the anti-aliasing filter. The passband must therefore fit below the Nyquist frequency of both the recording and the target rate:

| Constraint | Rule |
|:-----------|:-----|
| Nyquist | `highcut_freq` must be strictly below half of the smaller of the current sampling rate and `sampling_rate` |
| Ordering | `lowcut_freq` must be below `highcut_freq` |

Either violation raises a `ValueError` before any data are filtered, naming the offending value and how to resolve it (lower `highcut_freq`, or raise `sampling_rate` above twice `highcut_freq`).

{: .important }
> Because the check also covers the downsampling target, a 40 Hz low-pass combined with `sampling_rate = 80` is rejected even when the original recording was sampled at 1000 Hz.

### Example Configuration

```ini
[preprocessing_config]
temporal_filter_data = True
filter_method = fir
lowcut_freq = 2
highcut_freq = 20
```

---

## Spatial Filtering

### Purpose

Optional spatial enhancement replaces each electrode's signal with the average of that electrode and its nearest neighbors. This can improve topographic pattern recognition by:

- Reducing localized electrode noise
- Increasing signal-to-noise ratio
- Improving template identification accuracy

Neighborhoods are derived from the 3D electrode positions in the montage. For each electrode, COMET sorts the other electrodes by distance and looks for a natural gap in that distance distribution, keeping between 3 and 8 neighbors. The neighborhood size is chosen automatically and is not exposed as a configuration parameter.

### Trade-offs

| Benefit | Risk |
|:--------|:-----|
| Enhanced template identification | Blurred topographic boundaries |
| Reduced sensitivity to noisy channels | Reduced detection of rapid transitions |
| Improved clustering stability | Artificial smoothing of genuine patterns |

### When to Use

| Scenario | Recommendation |
|:---------|:---------------|
| High-quality recordings | Leave disabled (preserve resolution) |
| Moderate to high channel noise | Enable, then compare topographies before and after |
| Low electrode density (<32 channels) | **Do not use** (insufficient sampling) |

{: .warning }
> Spatial filtering is **not recommended** for datasets with low electrode density, as insufficient spatial sampling can prevent accurate interpolation and may produce false topographic features.

### Configuration Parameters

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `spatial_filter_data` | Enable neighbor averaging | `False` | `True`, `False` |

### Example Configuration

```ini
[preprocessing_config]
spatial_filter_data = True
```

{: .note }
> The `smoothing_gfp` and `smoothing_distance` parameters in `[clustering_config]` are unrelated to spatial filtering; they constrain GFP peak detection during [data selection]({% link modules/data-selection.md %}).

---

## Downsampling

### Purpose

Reduce computational load while preserving microstate-relevant frequencies.

### Considerations

| Factor | Guideline |
|:-------|:----------|
| **Nyquist criterion** | Sample rate ≥ 2× highest frequency of interest |
| **Microstate timing** | Minimum ~4 ms resolution for typical durations |
| **Memory constraints** | Lower rates reduce memory usage |

### Recommended Rates

| Original Rate | Target Rate | Suitable For |
|:--------------|:------------|:-------------|
| 1000+ Hz | 250 Hz | Standard analysis |
| 500 Hz | 250 Hz | Standard analysis |
| 250 Hz | Keep original | Already optimal |
| 128 Hz | Keep original | May limit high-frequency content |

### Configuration Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `downsample_data` | Enable downsampling | `True` |
| `sampling_rate` | Target sample rate (Hz) | `250` |

---

## Channel Management

### Removing Channels

Remove non-EEG or problematic channels before analysis:

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `remove_channels` | Enable channel removal | `False` |
| `channels_to_remove` | Channels to remove | `[]` |

**Example:**
```ini
[preprocessing_config]
remove_channels = True
channels_to_remove = ECG,EMG,EOG1,EOG2
```

### Handling Missing Channels

If `remove_channels = False`, EEG-COMET automatically removes channels that are missing in any dataset to ensure consistency.

---

## Complete Configuration Example

### Standard Resting-State Analysis

```ini
[io_config]
# Data handling
load_all_files = True
data_type = raw

[preprocessing_config]
# Temporal filtering
temporal_filter_data = True
filter_method = fir
lowcut_freq = 2
highcut_freq = 20

# Downsampling
downsample_data = True
sampling_rate = 250

# Spatial filtering
spatial_filter_data = False

# Channel management
remove_channels = False
channels_to_remove = []
```

### Event-Related Analysis

```ini
[io_config]
# Data handling
load_all_files = True
data_type = epoched

[preprocessing_config]
# Temporal filtering (broader band)
temporal_filter_data = True
filter_method = fir
lowcut_freq = 1
highcut_freq = 40

# Keep original sampling for timing precision
downsample_data = False
sampling_rate = 500

# Remove non-EEG channels
remove_channels = True
channels_to_remove = HEOG,VEOG,ECG
```

---

## Quality Assessment

After preprocessing, verify data quality:

### Visual Inspection

1. **Time series** - Check for residual artifacts
2. **Power spectrum** - Verify filter effects
3. **Topographies** - Ensure realistic spatial patterns

### Automated Checks

EEG-COMET performs:

| Check | Purpose |
|:------|:--------|
| Amplitude range | Flag extreme values |
| Flatline detection | Identify dead channels |
| Noise levels | Estimate signal quality |

---

## Best Practices

1. **Match filter to analysis goals**
   - Broad band (1-40 Hz) for event-related
   - Standard band (2-20 Hz) for resting-state

2. **Use FIR for event-related paradigms**
   - Better temporal precision
   - Linear phase response

3. **Be conservative with spatial smoothing**
   - Start with no smoothing
   - Add only if needed for stability

4. **Document all parameters**
   - EEG-COMET logs all settings
   - Essential for reproducibility

5. **Verify preprocessing effects**
   - Compare before/after
   - Check for introduced artifacts

---

## Common Issues

<div class="callout warning">
<strong>Edge artifacts after filtering</strong><br>
FIR filters create edge effects. Ensure recordings have sufficient padding at start/end, or trim edges after filtering.
</div>

<div class="callout warning">
<strong>Filtering rejected with a Nyquist error</strong><br>
<code>highcut_freq</code> is at or above half of either the recording's sampling rate or the downsampling target. Lower <code>highcut_freq</code>, or raise <code>sampling_rate</code> above twice <code>highcut_freq</code>.
</div>

<div class="callout warning">
<strong>Overly smooth topographies</strong><br>
Disable spatial filtering. High electrode density data often needs no smoothing.
</div>

---

## References

- Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: A review. *NeuroImage*, 180, 577–593. [https://doi.org/10.1016/j.neuroimage.2017.11.062](https://doi.org/10.1016/j.neuroimage.2017.11.062)
- Khanna, A., Pascual-Leone, A., Michel, C. M., & Farzan, F. (2015). Microstates in resting-state EEG: Current status and future directions. *Neuroscience & Biobehavioral Reviews*, 49, 105–113. [https://doi.org/10.1016/j.neubiorev.2014.12.010](https://doi.org/10.1016/j.neubiorev.2014.12.010)

---

## Next Step

[**Clustering Data Selection Module →**]({% link modules/data-selection.md %})

