---
title: 8. Feature Extraction
layout: default
parent: Modules
nav_order: 8
description: "Feature Extraction Module - Microstate metrics and dynamics"
---

# Feature Extraction Module
{: .no_toc }

Classical temporal metrics, complexity measures, and event-related dynamics.
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

EEG-COMET provides a comprehensive suite of microstate metrics spanning:

1. **Classical Temporal Metrics** - Standard microstate parameters
2. **Advanced Complexity Measures** - Sequence analysis
3. **Transition Probabilities** - Sequential dependencies
4. **Event-Related Dynamics** - Trial-level analysis

---

## Classical Temporal Metrics

### Coverage (COV)

**Proportional temporal dominance** of each microstate.

$$\text{COV}_k = \frac{\text{Total time in microstate } k}{\text{Total recording time}}$$

| Property | Value |
|:---------|:------|
| Range | 0 to 1 (or 0-100%) |
| Sum across classes | 1.0 (100%) |
| Interpretation | Relative temporal prevalence |

### Occurrence (OCC)

**Frequency of microstate appearances** per unit time.

$$\text{OCC}_k = \frac{\text{Number of microstate } k \text{ segments}}{\text{Recording duration (seconds)}}$$

| Property | Value |
|:---------|:------|
| Units | Occurrences per second (Hz) |
| Typical range | 2-8 Hz |
| Interpretation | How often state appears |

### Mean Duration (DUR)

**Average temporal stability** before transitioning.

For a microstate $k$ with $N_k$ contiguous segments and per-segment lengths $\ell_{k,1}, \dots, \ell_{k,N_k}$ (in samples), the per-segment lengths are summarised into a single value (in milliseconds) using the configurable `duration_method` parameter.

| Property | Value |
|:---------|:------|
| Units | Milliseconds (ms) |
| Typical range | 40-120 ms |
| Interpretation | Temporal persistence |

#### Aggregation methods (`duration_method`)

EEG-COMET supports four ways of summarising the per-segment run lengths. All return the same units (ms) but differ in how they handle the long-tail distribution that is typical of high-coverage microstates.

| Method | Formula | Notes |
|:-------|:--------|:------|
| `geometric` (**default**) | $\text{DUR}_k = \exp\!\left(\frac{1}{N_k}\sum_i \ln \ell_{k,i}\right) \cdot \frac{1000}{f_s}$ | Geometric mean of run lengths. Robust to long-tail outliers that otherwise inflate the arithmetic mean for dominant microstates. |
| `arithmetic` | $\text{DUR}_k = \left(\bar{\ell}_k - 1\right) \cdot \frac{1000}{f_s}$ | Mean of run lengths converted with the $(N-1)/f_s$ interval convention. Algebraically consistent with COV and OCC (see relationship below). |
| `median` | $\text{DUR}_k = \operatorname{median}(\ell_{k,i}) \cdot \frac{1000}{f_s}$ | Robust central-tendency estimator. |
| `trimmed_mean` | 10% symmetric trimmed mean of $\ell_{k,i}$, then $\cdot\,\frac{1000}{f_s}$ | Falls back to the arithmetic mean when fewer than 11 segments are available. |

{: .note }
> The default switched from `arithmetic` to `geometric` because heavy-tailed run-length distributions on dominant microstates (e.g. when long quiet segments are present) make the arithmetic mean over-estimate the typical persistence. The geometric mean tracks the bulk of the distribution far better.

### Relationship

When `duration_method = arithmetic`, the three classical metrics are algebraically consistent:

$$\text{COV}_k \;=\; \left(\text{DUR}_k + \tfrac{1000}{f_s}\right) \cdot \text{OCC}_k \;\big/\; 1000$$

For the other aggregation methods (`geometric`, `median`, `trimmed_mean`) DUR is no longer the arithmetic mean of segment lengths, so the identity $\text{COV}_k = \text{DUR}_k \times \text{OCC}_k$ holds only approximately. Use `arithmetic` if you need DUR, COV and OCC to be exactly self-consistent (e.g. for analytical derivations).

### Global Explained Variance (GEV)

**Proportion of total topographic variance** explained by each template.

$$\text{GEV}_k = \frac{\sum_{t \in k} r_{t,k}^2 \cdot \text{GFP}_t^2}{\sum_{t} \text{GFP}_t^2}$$

| Property | Value |
|:---------|:------|
| Range | 0 to 1 |
| Sum across classes | Total GEV |
| Interpretation | Template explanatory power |

---

## Transition Probabilities (TP)

### Definition

First-order sequential dependencies between microstate pairs.

$$\text{TP}_{i \to j} = P(\text{next state} = j \mid \text{current state} = i)$$

### Matrix Structure

For K microstates, produces K×K transition matrix:

|  | To A | To B | To C | To D |
|:---|:---:|:---:|:---:|:---:|
| **From A** | — | 0.35 | 0.40 | 0.25 |
| **From B** | 0.30 | — | 0.45 | 0.25 |
| **From C** | 0.35 | 0.40 | — | 0.25 |
| **From D** | 0.33 | 0.33 | 0.34 | — |

### Self-Transitions

By convention, self-transitions (A→A) are excluded or set to zero, as consecutive samples of the same microstate are part of the same segment.

### Interpretation

| Pattern | Meaning |
|:--------|:--------|
| High TP(A→B) | State A preferentially leads to B |
| Symmetric TP | Similar forward/backward probabilities |
| Asymmetric TP | Directional sequence preferences |

---

## Advanced Complexity Measures

### Entropy Rate (ER)

**Information content conditional on sequential history.**

$$H_r = -\sum_i p_i \sum_j p_{j|i} \log_2 p_{j|i}$$

| Property | Interpretation |
|:---------|:---------------|
| Low entropy | Predictable sequences |
| High entropy | Random/unpredictable sequences |
| Range | 0 to log₂(K) |

### Lempel-Ziv Complexity (LZC)

**Algorithmic assessment of sequence diversity.**

Counts the number of distinct patterns in the microstate sequence, normalized by sequence length.

| Property | Interpretation |
|:---------|:---------------|
| Low LZC | Repetitive patterns |
| High LZC | Diverse, complex sequences |
| Theoretical max | Random sequence |

### Hurst Exponent (HE)

**Long-range temporal correlations** across multiple timescales.

| Value | Pattern Type |
|:------|:-------------|
| H = 0.5 | Random walk (no memory) |
| H > 0.5 | Persistent (positive autocorrelation) |
| H < 0.5 | Anti-persistent (negative autocorrelation) |

### Entropy Representation Ratio (ERR)

**Observed sequence organization relative to random transitions.**

$$\text{ERR} = \frac{H_{\text{observed}}}{H_{\text{random}}}$$

| Value | Interpretation |
|:------|:---------------|
| ERR = 1 | Random-like organization |
| ERR < 1 | More structured than random |
| ERR > 1 | More chaotic than expected |

---

## Analysis Modes

### Static Mode

Single feature value computed over entire recording.

**Use case:** Traditional resting-state analysis, group comparisons.

```ini
feature_mode = static
```

### Windowed Mode

Features computed within sliding windows to track temporal changes.

**Use case:** Continuous recordings with alternating conditions.

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `window_size` | Window duration (seconds) | `1` |
| Window overlap | Typically 50% | Configurable |

```ini
feature_mode = windowed
window_size = 2
```

**Advantages:**
- Preserves data continuity
- Captures rapid changes
- Avoids artificial segment boundaries

### Event-Related Mode

Trial-level analysis around experimental events.

**Use case:** ERP paradigms, TMS-EEG, cognitive tasks.

---

## Event-Related Features

### Trial-Level Pre/Post Features

For each trial, compute features in:
- **Pre-event window:** -1000 to -10 ms (baseline)
- **Post-event window:** +10 to +1000 ms (response)

| Extracted Features |
|:-------------------|
| COV, OCC, DUR, GEV per microstate |
| Window customizable by user |

{: .note }
> Complexity measures (ER, LZC, HE) are typically excluded from short windows as they may not yield reliable estimates.

### Relative Occurrence Frequency (ROF)

**Time-resolved microstate presence across trials.**

```
For each timepoint t after event:
    1. Count microstate presence across all trials
    2. Apply centered log-ratio transformation
    3. Subtract baseline median
    
Result: ROF(t) for each microstate class
```

| Feature | Description |
|:--------|:------------|
| Time resolution | Sample-by-sample |
| Baseline correction | Pre-event median subtracted |
| Normalization | Log-ratio for compositional data |

### Relative Transition Frequency (RTF)

**Event-related changes in transition probabilities.**

Measures how inter-microstate transitions systematically change following experimental events, indicating reorganization of network communication.

---

## Configuration

### Parameters

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `export_format` | Output file format | `.csv` | `.csv`, `.pkl`, `.hdf`, `.json` |
| `feature_list` | Features to extract | `COV,OCC,MMD` | See below |
| `feature_mode` | Analysis mode | `static` | `static`, `windowed`, `event_related` |
| `window_size` | Window duration (s) | `1` | 0.5-10 |
| `feature_types` | Comparison types | `real,surrogate,random` | See below |
| `duration_method` | DUR aggregation | `geometric` | `geometric`, `arithmetic`, `median`, `trimmed_mean` |

### Feature List Options

| Code | Feature |
|:-----|:--------|
| `COV` | Coverage |
| `OCC` | Occurrence |
| `MMD` | Mean Duration |
| `DUR` | Duration (alias for MMD) |
| `TP` | Transition Probabilities |
| `GEV` | Global Explained Variance |
| `LZC` | Lempel-Ziv Complexity |
| `ER` | Entropy Rate |
| `HE` | Hurst Exponent |
| `ERR` | Entropy Representation Ratio |

### Feature Types

| Type | Description |
|:-----|:------------|
| `real` | Actual observed values |
| `surrogate` | Time-shuffled control |
| `random` | Random sequence baseline |

### Example Configurations

#### Standard Resting-State

```ini
[features_config]
export_format = .csv
feature_list = COV,OCC,MMD,GEV,TP
feature_mode = static
feature_types = real
duration_method = geometric
```

#### Comprehensive Analysis

```ini
[features_config]
export_format = .csv
feature_list = COV,OCC,MMD,GEV,TP,LZC,ER,HE
feature_mode = static
feature_types = real,surrogate
duration_method = geometric
```

#### COV-DUR-OCC Algebraic Consistency

Use the arithmetic aggregation when DUR must satisfy `COV ≈ DUR × OCC` exactly (e.g. when reporting all three metrics in tables that should add up):

```ini
[features_config]
feature_list = COV,OCC,MMD
feature_mode = static
duration_method = arithmetic
```

#### Windowed Analysis

```ini
[features_config]
export_format = .csv
feature_list = COV,OCC,MMD,GEV
feature_mode = windowed
window_size = 2
feature_types = real
```

---

## Output Files

### CSV Format

Standard comma-separated values:

```csv
subject,condition,microstate,COV,OCC,DUR,GEV
sub-01,rest,A,0.28,3.2,87.5,0.22
sub-01,rest,B,0.24,2.9,82.7,0.18
...
```

### Transition Matrix

Separate file with full transition probabilities:

```csv
from,to_A,to_B,to_C,to_D
A,0.0,0.35,0.40,0.25
B,0.30,0.0,0.45,0.25
...
```

### Windowed Output

Time-indexed features:

```csv
subject,window_start,window_end,microstate,COV,OCC,DUR
sub-01,0,2000,A,0.30,3.5,85.7
sub-01,1000,3000,A,0.27,3.1,87.1
...
```

---

## Best Practices

1. **Select appropriate features for research question**
   - Basic: COV, OCC, DUR
   - Sequence analysis: TP, ER, LZC
   - Dynamics: HE, ERR

2. **Match mode to experimental design**
   - Static for between-subject comparisons
   - Windowed for within-session dynamics
   - Event-related for trial-level analysis

3. **Include surrogate comparisons**
   - Validates observed patterns
   - Provides null distribution

4. **Report all extracted features**
   - Avoid selective reporting
   - Include in supplementary materials

5. **Consider multiple export formats**
   - CSV for general use
   - HDF for large datasets

---

## Next Step

[**Statistical Analysis Module →**]({% link modules/statistical-analysis.md %})

