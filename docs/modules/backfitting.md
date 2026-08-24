---
title: 7. Backfitting
layout: default
parent: Modules
nav_order: 7
description: "Template Backfitting Module - Segmentation and refinement"
---

# Template Backfitting and Segmentation Refinement Module
{: .no_toc }

Template assignment with segment duration optimization and refinement strategies.
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

After identifying and labeling microstates, each time point in the continuous EEG recording is assigned to the template with the highest spatial correlation through a **backfitting** process (Pascual-Marqui et al., 1995; Brunet et al., 2011). This produces a preliminary segmentation that often requires refinement to handle implausibly short segments.

### Two Key Decisions

1. **Minimum segment duration threshold** - How short is too short?
2. **Strategy for handling short segments** - What to do with them?

---

## Backfitting Process

### Algorithm

```
For each timepoint t:
    1. Calculate spatial correlation with each template
    2. Apply polarity invariance (absolute correlation)
    3. Assign to template with highest correlation
    4. Record assignment and correlation strength
```

### Backfitting Modes

| Mode | Description | Use Case |
|:-----|:------------|:---------|
| `all` | Assign all timepoints | Standard analysis |
| `peaks` | Assign only GFP peaks, then extend each label across the surrounding GFP trough-to-trough interval | Traditional approach |

{: .note }
> Correlation thresholding and segment refinement apply to the `all` mode. In `peaks` mode the label of each GFP peak already spans a full trough-to-trough interval, so no additional filtering is performed.

### Correlation Quality Control

Optionally, timepoints whose absolute correlation with their assigned template is too weak can be rejected instead of labeled. Set `min_correlation_threshold` to the minimum acceptable absolute correlation (typically 0.5 for a liberal criterion up to 0.7 for a conservative one), or leave it at `False` to label every timepoint.

Rejected timepoints are marked as unassigned and **stay unassigned** through every subsequent refinement step. Length-preserving strategies fill only the gaps left by short-segment rejection; they never re-admit a timepoint that failed the correlation threshold. The threshold is therefore always honoured, regardless of the `filter_segments_option` in use.

### Initial Segmentation

The raw backfitting produces many very short segments:

- Segments < 10 ms are common
- Typical stable microstates last > 40 ms
- Brief segments likely represent noise or transitions

{: .note }
> Empirical data suggests typical microstate durations of 40-120 ms. Segments far below this range warrant scrutiny (Koenig et al., 2002; Michel & Koenig, 2018).

---

## Minimum Segment Duration

### The Problem

Extremely short microstate segments (< 10 ms) likely represent:
- Measurement noise
- Transitional phases
- Misclassifications

Rather than genuine microstate changes.

The threshold is specified in milliseconds and converted to samples using the sampling rate. A segment counts as short when its duration is **less than or equal to** the threshold; only strictly longer segments survive filtering.

### Selection Methods

#### Manual Input

Specify a fixed threshold based on:
- Prior literature (often 20-30 ms)
- Sampling rate considerations
- Theoretical expectations

#### Automated Optimization

Setting `identify_short_window` lets EEG-COMET determine the threshold automatically:

```
For each candidate threshold from 2 ms (or one sample, whichever
is longer) up to 60 ms:
    1. Calculate combined quality score:
       - Topographic correspondence (60% weight)
       - Temporal stability (30% weight)
       - Data coverage (10% weight)
    2. Track optimal threshold

Select threshold maximizing quality score,
then clamp the result to the 5-50 ms range
```

Up to 20 evenly spaced candidates are tested. If fewer than 20 sample steps fit into that span, every sample step is tested instead.

### Automated Lambda Optimization

When `identify_short_window` is enabled and `filter_segments_option = smooth`, the non-smoothness penalty is optimized as well. Five candidate values spanning **5 to 15** are smoothed with the configured half-window, scored with the same combined quality metric, and the best value is kept.

### Multi-Recording Optimization

When analyzing multiple recordings:
1. Optimal threshold determined for each recording
2. Median across recordings is selected
3. Rounded to nearest sampling interval

The optimal lambda is pooled the same way, taking the median across recordings rounded to one decimal place. This provides robust central estimates resistant to outliers.

---

## Short Segment Handling Strategies

### Strategy Categories

| Category | Strategies | Temporal Structure |
|:---------|:-----------|:-------------------|
| **Data Reduction** | Segment Exclusion | Gaps created |
| **Length-Preserving** | Local Context, Symmetric Extension, Sequential Smoothing | Maintained |

Length-preserving strategies fill only the gaps created by short-segment rejection. Timepoints rejected by `min_correlation_threshold` remain unassigned under every strategy.

---

### Strategy 1: Segment Exclusion (`remove`)

**Remove short segments entirely, creating unlabeled gaps.**

```
Before: A A A [B] A A A A A
After:  A A A [ ] A A A A A  (gap where B was)
```

| Pros | Cons |
|:-----|:-----|
| Simple, clean | Disrupts temporal continuity |
| Removes noise | Unsuitable for event-related |
| Good for summary statistics | Loses information |

**Best for:** Resting-state analysis where continuous labeling isn't required.

---

### Strategy 2: Local Context Reassignment (`replace_high`)

**Replace the short segment with whichever adjoining segment has the longer run.**

The full run lengths of the segments immediately before and after the gap are compared, and the entire gap is assigned to the longer of the two. If the two runs are equally long, the preceding label wins. When only one side has a valid label, the gap takes that label.

```
Before: A A A [B] A A A A A
After:  A A A [A] A A A A A  (B replaced by longer-running neighbor A)
```

| Pros | Cons |
|:-----|:-----|
| Preserves temporal length | May over-extend dominant states |
| Simple rule | Local bias |
| Maintains continuity | Ignores temporal direction |

**Best for:** When local dominance is a reasonable assumption.

---

### Strategy 3: Symmetric Extension (`replace_half`)

**Divide short segment between preceding and following microstates.**

The gap is split down the middle, with an odd extra sample going to the preceding label. When only one side has a valid label, the whole gap takes that label.

```
Before: A A [B B] C C C C
After:  A A [A C] C C C C  (first half to A, second half to C)
```

| Pros | Cons |
|:-----|:-----|
| Balanced approach | Artificial boundary placement |
| No directional bias | May create unrealistic transitions |
| Preserves overall timing | Splits continuous segments |

**Best for:** When no clear preference for direction exists.

---

### Strategy 4: Sequential Smoothing (`smooth`)

**Apply temporal smoothing considering both past and future context.** This implements the window-based smoothing algorithm of Pascual-Marqui et al. (1995), where the window half-width `half_window_size` and the non-smoothness penalty `lambda` control the trade-off between local context and label stability.

```
Algorithm:
1. Count how often each label occurs in the window [t-b, t+b],
   excluding t itself (an unweighted count of neighbors)
2. Subtract lambda x count from the spatial misfit of each
   candidate label, and take the label with the lowest total cost
3. Iterate until no assignment changes or GEV converges
4. Reject any remaining short segments and redistribute them to
   the best-correlating neighboring template
```

The window is unweighted: every neighbor inside `[t-b, t+b]` contributes equally to the count, irrespective of its distance from `t`.

In step 4, each rejected stretch is split between its two neighboring templates at the point where the opposite template becomes the better spatial fit, and any ambiguous remainder in the middle is divided evenly. The step repeats until no segment shorter than or equal to the threshold remains, since redistribution can itself create new short segments at the split boundaries.

| Pros | Cons |
|:-----|:-----|
| Principled statistical approach | More complex |
| Considers full local context | Additional parameters |
| Smooths noise while preserving transitions | May over-smooth |

**Best for:** Sophisticated analysis requiring temporal coherence.

### Smoothing Parameters

| Parameter | Description | Default | Range |
|:----------|:------------|:--------|:------|
| `convergence_epsilon` | Convergence criterion | `1e-6` | 1e-8 to 1e-4 |
| `half_window_size` | Half-window b (samples) | `3` | 1-10 |
| `smoothness_penalty` | Non-smoothness penalty (lambda) | `5` | 1-20 |

`half_window_size` is the half-window b of Pascual-Marqui et al. (1995): the smoothing window spans `[t-b, t+b]`, so b = 3 covers 7 samples (28 ms at 250 Hz). It is an independent parameter and is not derived from `filter_segments_less_than`.

---

## Strategy Selection Guide

| Research Context | Recommended Strategy |
|:-----------------|:---------------------|
| Resting-state, summary statistics | Segment Exclusion |
| Resting-state, sequence analysis | Sequential Smoothing |
| Event-related, timing critical | Sequential Smoothing |
| Quick exploration | Local Context |
| Balanced approach | Symmetric Extension |

---

## Configuration

### Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `backfit_to` | What to backfit | `all` |
| `identify_short_window` | Auto-optimize threshold (and lambda, when smoothing) | `False` |
| `filter_segments` | Apply segment refinement | `False` |
| `filter_segments_less_than` | Reject segments lasting at most this long (ms) | `20` |
| `filter_segments_option` | Handling strategy | `smooth` |
| `min_correlation_threshold` | Minimum absolute correlation to keep a timepoint (`False` disables) | `False` |

### Example Configurations

#### Resting-State Analysis

```ini
[backfitting_config]
backfit_to = all
identify_short_window = False
filter_segments = True
filter_segments_less_than = 20
filter_segments_option = remove
```

#### Event-Related Analysis

```ini
[backfitting_config]
backfit_to = all
identify_short_window = True
filter_segments = True
filter_segments_less_than = 20
filter_segments_option = smooth
convergence_epsilon = 1e-6
half_window_size = 3
smoothness_penalty = 5
```

#### Conservative (Minimal Processing)

```ini
[backfitting_config]
backfit_to = all
identify_short_window = False
filter_segments = False
```

---

## Output

### Segmentation Results

For each recording, backfitting produces:

| Output | Description |
|:-------|:------------|
| `labels` | Microstate assignment per timepoint |
| `correlations` | Assignment confidence per timepoint |
| `segments` | Start/end indices for each segment |
| `durations` | Duration of each segment |

### Quality Metrics

| Metric | Description |
|:-------|:------------|
| **Mean segment duration** | Average stability |
| **Segment count** | Number of transitions |
| **Assignment confidence** | Average correlation with templates |

---

## Visualization

### Segmentation View

EEG-COMET displays:
- Color-coded timeline of microstate assignments
- Segment boundaries
- Correlation strength overlay

### Before/After Comparison

View the effect of segment refinement:
- Original (noisy) segmentation
- Refined segmentation
- Highlight removed/changed segments

---

## Best Practices

1. **Choose strategy based on paradigm**
   - Event-related: Length-preserving
   - Resting-state: Either approach

2. **Use automated threshold selection**
   - More objective than manual
   - Data-driven optimization

3. **Document all parameters**
   - Critical for reproducibility
   - Include in methods section

4. **Validate segmentation quality**
   - Check mean durations
   - Compare to expected ranges

5. **Be consistent within studies**
   - Same parameters for all subjects
   - Same strategy throughout

---

## Troubleshooting

<div class="callout warning">
<strong>Too many short segments after refinement</strong><br>
Increase minimum duration threshold. Consider if K is too high (too many similar templates).
</div>

<div class="callout warning">
<strong>Mean duration seems too long</strong><br>
May indicate over-smoothing. Reduce smoothness_penalty parameter or try different strategy.
</div>

<div class="callout warning">
<strong>Large gaps in segmentation (remove strategy)</strong><br>
This is expected with strict thresholds. If problematic, switch to length-preserving strategy.
</div>

<div class="callout warning">
<strong>Smoothing doesn't converge</strong><br>
Increase convergence_epsilon. Smoothing stops after at most 20 passes in any case, so also check for data quality issues.
</div>

---

## References

- Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of brain electrical activity into microstates: model estimation and validation. *IEEE Transactions on Biomedical Engineering*, 42(7), 658–665. [https://doi.org/10.1109/10.391164](https://doi.org/10.1109/10.391164)
- Brunet, D., Murray, M. M., & Michel, C. M. (2011). Spatiotemporal analysis of multichannel EEG: CARTOOL. *Computational Intelligence and Neuroscience*, 2011, 813870. [https://doi.org/10.1155/2011/813870](https://doi.org/10.1155/2011/813870)
- Koenig, T., Prichep, L., Lehmann, D., Valdes Sosa, P., Braeker, E., Kleinlogel, H., Isenhart, R., & John, E. R. (2002). Millisecond by millisecond, year by year: Normative EEG microstates and developmental stages. *NeuroImage*, 16(1), 41–48. [https://doi.org/10.1006/nimg.2002.1070](https://doi.org/10.1006/nimg.2002.1070)
- Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: A review. *NeuroImage*, 180, 577–593. [https://doi.org/10.1016/j.neuroimage.2017.11.062](https://doi.org/10.1016/j.neuroimage.2017.11.062)

---

## Next Step

[**Feature Extraction Module →**]({% link modules/feature-extraction.md %})

