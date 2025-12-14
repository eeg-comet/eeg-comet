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
2. **Spatial Filtering** - K-nearest-neighbor smoothing

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
> The 2-20 Hz range is most commonly used in microstate research, highlighting key topographic patterns while reducing noise.

### Configuration Parameters

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `filter_data` | Enable temporal filtering | `True` | `True`, `False` |
| `filter_method` | Filter type | `fir` | `fir`, `iir` |
| `lowcut_freq` | High-pass cutoff (Hz) | `2` | 0.1 - 10 |
| `highcut_freq` | Low-pass cutoff (Hz) | `20` | 10 - 100 |

### Example Configuration

```ini
[preprocessing_config]
filter_data = True
filter_method = fir
lowcut_freq = 2
highcut_freq = 20
```

---

## Spatial Filtering

### Purpose

Optional spatial enhancement using k-nearest-neighbor interpolation can improve topographic pattern recognition by:

- Reducing localized electrode noise
- Increasing signal-to-noise ratio
- Improving template identification accuracy

### Trade-offs

| Benefit | Risk |
|:--------|:-----|
| Enhanced template identification | Blurred topographic boundaries |
| Reduced sensitivity to noisy channels | Reduced detection of rapid transitions |
| Improved clustering stability | Artificial smoothing of genuine patterns |

### When to Use

| Scenario | Recommendation |
|:---------|:---------------|
| High-quality recordings | Avoid smoothing (preserve resolution) |
| Moderate noise | Light smoothing (distance = 3-5) |
| Noisy datasets | Moderate smoothing (distance = 5-10) |
| Low electrode density (<32 channels) | **Do not use** (insufficient sampling) |

{: .warning }
> Spatial filtering is **not recommended** for datasets with low electrode density, as insufficient spatial sampling can prevent accurate interpolation and may produce false topographic features.

### Configuration Parameters

| Parameter | Description | Default | Range |
|:----------|:------------|:--------|:------|
| `smoothing_gfp` | Enable spatial smoothing | `True` | `True`, `False` |
| `smoothing_distance` | Number of neighbors | `10` | 3 - 20 |

### Example Configuration

```ini
[clustering_config]
smoothing_gfp = True
smoothing_distance = 10
```

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
| `sample_rate` | Target sample rate (Hz) | `250` |

---

## Channel Management

### Removing Channels

Remove non-EEG or problematic channels before analysis:

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `remove_channels` | Enable channel removal | `False` |
| `chan2rm` | Channels to remove | `[]` |

**Example:**
```ini
[preprocessing_config]
remove_channels = True
chan2rm = ECG,EMG,EOG1,EOG2
```

### Handling Missing Channels

If `remove_channels = False`, EEG-COMET automatically removes channels that are missing in any dataset to ensure consistency.

---

## Complete Configuration Example

### Standard Resting-State Analysis

```ini
[preprocessing_config]
# Data handling
load_all_files = True
datatype = raw

# Temporal filtering
filter_data = True
filter_method = fir
lowcut_freq = 2
highcut_freq = 20

# Downsampling
downsample_data = True
sample_rate = 250

# Channel management
remove_channels = False
chan2rm = []
```

### Event-Related Analysis

```ini
[preprocessing_config]
# Data handling
load_all_files = True
datatype = epoched

# Temporal filtering (broader band)
filter_data = True
filter_method = fir
lowcut_freq = 1
highcut_freq = 40

# Keep original sampling for timing precision
downsample_data = False
sample_rate = 500

# Remove non-EEG channels
remove_channels = True
chan2rm = HEOG,VEOG,ECG
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
<strong>Unexpected spectral content</strong><br>
Check that filter frequencies are appropriate for your sampling rate. Low-pass should be well below Nyquist frequency.
</div>

<div class="callout warning">
<strong>Overly smooth topographies</strong><br>
Reduce or disable spatial smoothing. High electrode density data often needs no smoothing.
</div>

---

## Next Step

[**Clustering Data Selection Module →**]({% link modules/data-selection.md %})

