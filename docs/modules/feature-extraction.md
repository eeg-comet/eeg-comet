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

The classical microstate parameters (coverage, occurrence, mean duration, and global explained variance) follow the definitions established in the foundational and normative microstate literature (Lehmann et al., 1987; Koenig et al., 2002; Michel & Koenig, 2018; Khanna et al., 2015).

### Coverage (COV)

**Proportional temporal dominance** of each microstate.

$$\text{COV}_k = 100 \cdot \frac{\text{Samples labelled } k}{\text{Analysable samples}}$$

Rejected timepoints — those carrying the `NaN` label because they failed `min_correlation_threshold` or the short-segment filter — are excluded from both the numerator and the denominator. COV therefore expresses the percentage of *analysable* time rather than of wall-clock recording time, and sums to 100% across the real microstate classes.

| Property | Value |
|:---------|:------|
| Units | Percent (0-100) |
| Sum across classes | 100% |
| Interpretation | Relative temporal prevalence |

### Occurrence (OCC)

**Frequency of microstate appearances** per unit time.

$$\text{OCC}_k = \frac{\text{Number of microstate } k \text{ segments}}{\text{Analysable duration (seconds)}}$$

A segment is a contiguous run of identical labels. In averaged mode the denominator counts only analysable (non-rejected) samples; in sliding mode it is the nominal window length in seconds. A stretch of rejected samples still acts as a boundary, so two runs of state $$k$$ separated by rejected timepoints are counted as two segments rather than merged into one.

| Property | Value |
|:---------|:------|
| Units | Occurrences per second (Hz) |
| Typical range | 2-8 Hz |
| Interpretation | How often state appears |

### Mean Duration (DUR)

**Average temporal stability** before transitioning.

For a microstate $$k$$ with $$N_k$$ contiguous segments and per-segment lengths $$\ell_{k,1}, \dots, \ell_{k,N_k}$$ (in samples), the per-segment lengths are summarised into a single value (in milliseconds) using the configurable `duration_method` parameter.

| Property | Value |
|:---------|:------|
| Units | Milliseconds (ms) |
| Typical range | 40-120 ms |
| Interpretation | Temporal persistence |

#### Aggregation methods (`duration_method`)

EEG-COMET supports four ways of summarising the per-segment run lengths. All return the same units (ms) but differ in how they handle the long-tail distribution that is typical of high-coverage microstates.

| Method | Formula | Notes |
|:-------|:--------|:------|
| `geometric` (**default**) | $$\text{DUR}_k = \exp\!\left(\frac{1}{N_k}\sum_i \ln \ell_{k,i}\right) \cdot \frac{1000}{f_s}$$ | Geometric mean of run lengths. Robust to long-tail outliers that otherwise inflate the arithmetic mean for dominant microstates. |
| `arithmetic` | $$\text{DUR}_k = \left(\bar{\ell}_k - 1\right) \cdot \frac{1000}{f_s}$$ | Mean of run lengths converted with the $$(N-1)/f_s$$ interval convention. Algebraically consistent with COV and OCC (see relationship below). |
| `median` | $$\text{DUR}_k = \operatorname{median}(\ell_{k,i}) \cdot \frac{1000}{f_s}$$ | Robust central-tendency estimator. |
| `trimmed_mean` | Mean of $$\ell_{k,i}$$ after discarding the $$\lfloor N_k/10 \rfloor$$ smallest and largest run lengths (at least one from each end), then $$\cdot\,\frac{1000}{f_s}$$ | With 10 or fewer segments no trimming is possible, so the plain mean of the run lengths is used (without the $$(N-1)$$ correction applied by `arithmetic`). |

{: .note }
> The default switched from `arithmetic` to `geometric` because heavy-tailed run-length distributions on dominant microstates (e.g. when long quiet segments are present) make the arithmetic mean over-estimate the typical persistence. The geometric mean tracks the bulk of the distribution far better.

### Relationship

When `duration_method = arithmetic`, the three classical metrics are algebraically consistent. With COV expressed as a percentage, DUR in milliseconds and OCC in Hz:

$$\text{COV}_k \;=\; \left(\text{DUR}_k + \tfrac{1000}{f_s}\right) \cdot \frac{\text{OCC}_k}{10}
\qquad\Longleftrightarrow\qquad
\text{DUR}_k \;=\; \frac{10 \cdot \text{COV}_k}{\text{OCC}_k} - \frac{1000}{f_s}$$

The other aggregation methods (`geometric`, `median`, `trimmed_mean`) do not report the arithmetic mean of segment lengths, so they are **not** algebraically tied to COV and OCC: the classical identity $$\text{DUR}_k = 1000 \cdot \text{COV}_k / (100 \cdot \text{OCC}_k)$$ does not hold, and the gap widens as the run-length distribution becomes more skewed. Use `arithmetic` if you need DUR, COV and OCC to be exactly self-consistent (e.g. for analytical derivations).

### Global Explained Variance (GEV)

**Proportion of total topographic variance** explained by each template.

$$\text{GEV}_k = 100 \cdot \frac{\sum_{t \in k} r_{t,k}^2 \cdot \text{GFP}_t^2}{\sum_{t} \text{GFP}_t^2}$$

| Property | Value |
|:---------|:------|
| Units | Percent (0-100) |
| Sum across classes | Total variance explained (%), normally well below 100% |
| Interpretation | Template explanatory power |

Because GEV is accumulated only over the timepoints assigned to each template, the per-state values sum to the total explained variance of the segmentation, not to 100%.

---

## Transition Probabilities (TP)

### Definition

First-order sequential dependencies between microstate pairs.

$$\text{TP}_{i \to j} = P(\text{next state} = j \mid \text{current state} = i) = \frac{n_{i \to j}}{\sum_{m \neq i} n_{i \to m}}$$

Each row is normalised on its own counts, so the probabilities leaving any given state sum to 1 regardless of how often that state is visited.

### Matrix Structure

For K microstates, produces K×K transition matrix whose off-diagonal entries sum to 1 along every row:

|  | To A | To B | To C | To D |
|:---|:---:|:---:|:---:|:---:|
| **From A** | — | 0.35 | 0.40 | 0.25 |
| **From B** | 0.30 | — | 0.45 | 0.25 |
| **From C** | 0.35 | 0.40 | — | 0.25 |
| **From D** | 0.33 | 0.33 | 0.34 | — |

### Self-Transitions

By convention, self-transitions (A→A) are excluded, as consecutive samples of the same microstate are part of the same segment. They are left out of the row totals, so removing them does not leak probability mass out of a row.

### Rejected Timepoints

Sample pairs where either side is a rejected timepoint (`NaN`) are ignored, so rejected samples never appear as a source or a target state. A rejected stretch sitting between two microstates breaks the link between them instead of producing a spurious A→B transition across the gap.

### Interpretation

| Pattern | Meaning |
|:--------|:--------|
| High TP(A→B) | State A preferentially leads to B |
| Symmetric TP | Similar forward/backward probabilities |
| Asymmetric TP | Directional sequence preferences |

---

## Advanced Complexity Measures

### Entropy Rate (ER)

**Information content conditional on sequential history** (von Wegner et al., 2017).

The entropy rate is estimated from how the joint (block) entropy grows with history length, rather than from a first-order Markov approximation. For each history length $$k$$ the joint entropy of all $$k$$-symbol words observed in the sequence is computed,

$$H(k) = -\sum_{w} p(w) \ln p(w)$$

and a straight line is fitted to $$H(k)$$ against $$k = 1, \dots, k_{\max}$$. The slope of that line is the entropy rate; the intercept is the excess entropy. `k_max` defaults to 6.

| Property | Interpretation |
|:---------|:---------------|
| Low entropy | Predictable sequences |
| High entropy | Random/unpredictable sequences |
| Units | Nats per sample (natural logarithm) |
| Range | 0 to ln(K) |

### Lempel-Ziv Complexity (LZC)

**Algorithmic assessment of sequence diversity** (Lempel & Ziv, 1976; von Wegner et al., 2017).

The sequence is first run-length collapsed, so each contiguous run of one label becomes a single symbol and LZC reflects the ordering of segments rather than their durations. The LZ76 parsing then counts the number of distinct patterns $$c$$ in the collapsed sequence of length $$n$$, normalised by the asymptotic maximum $$b = n / \log_2 n$$:

$$\text{LZC} = \frac{c}{n / \log_2 n}$$

Sequences that collapse to fewer than three symbols return 0.0, since LZ76 cannot be parsed and the normaliser is undefined.

| Property | Interpretation |
|:---------|:---------------|
| Low LZC | Repetitive patterns |
| High LZC | Diverse, complex sequences |
| Theoretical max | Random sequence |

### Hurst Exponent (HE)

**Long-range temporal correlations** across multiple timescales. Microstate sequences exhibit scale-free, long-range dependent dynamics (Van de Ville et al., 2010).

| Value | Pattern Type |
|:------|:-------------|
| H = 0.5 | Random walk (no memory) |
| H > 0.5 | Persistent (positive autocorrelation) |
| H < 0.5 | Anti-persistent (negative autocorrelation) |

### Entropy Representation Ratio (ERR)

**Distribution of the observed word inventory across entropy classes.**

Every word of `word_size` consecutive symbols (default 2) is read off the microstate sequence with a one-sample step. Each distinct word is scored by the Shannon entropy of its own symbol composition, and words that share an entropy value are grouped into an entropy class, numbered from the lowest entropy upwards. The exported value for each class is the share of all observed words that fall into it:

$$\text{ERR}_c = \frac{\text{Observed words in entropy class } c}{\text{Total observed words}}$$

Low-numbered classes hold repetitive words such as `AA`; higher classes hold words that mix distinct microstates. One column per class is written (`ERR_EntropyClass1`, `ERR_EntropyClass2`, …).

| Value | Interpretation |
|:------|:---------------|
| Mass in low classes | Sequence dwells in, and returns to, the same states |
| Mass in high classes | Sequence cycles through many distinct states |
| Sum across classes | 1.0 |

{: .note }
> In sliding mode ERR is reported differently: each window contributes the Shannon entropy (in nats) of its own label distribution rather than a per-class breakdown.

---

## Analysis Modes

### Averaged Mode

Single feature value computed over entire recording.

**Use case:** Traditional resting-state analysis, group comparisons.

```ini
feature_mode = averaged
```

### Sliding Mode

Features computed within consecutive windows to track temporal changes.

**Use case:** Continuous recordings with alternating conditions.

Windows are **non-overlapping** (tumbling): the recording is cut into back-to-back blocks of `sliding_window_size` seconds, and each block is analysed independently. There is no overlap parameter. A trailing block shorter than one full window is dropped, so the number of windows is $$\lfloor n_{\text{samples}} / (\texttt{sliding\_window\_size} \times f_s) \rfloor$$ and window $$i$$ spans $$[\,i \cdot \texttt{sliding\_window\_size},\ (i+1) \cdot \texttt{sliding\_window\_size}\,)$$ seconds.

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `sliding_window_size` | Window duration (seconds) | `1` |

```ini
feature_mode = sliding
sliding_window_size = 2
```

**Advantages:**
- Preserves data continuity
- Captures rapid changes
- Avoids artificial segment boundaries

{: .note }
> TP and LZC are not computed in sliding mode; request them with `feature_mode = averaged`.

### Pre/Post Event Mode

Trial-level analysis around experimental events.

**Use case:** ERP paradigms, TMS-EEG, cognitive tasks.

---

## Event-Related Features

### Trial-Level Pre/Post Features

For each trial, compute features in:
- **Pre-event window:** -1000 to -10 ms (baseline)
- **Post-event window:** +20 to +1000 ms (response)

| Extracted Features |
|:-------------------|
| Every selected feature except ROF and RTF, per microstate |
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
| `feature_list` | Features to extract | `OCC,DUR,COV` | See below |
| `feature_mode` | Analysis mode | `averaged` | `averaged`, `sliding`, `pre_post` |
| `sliding_window_size` | Window duration (s) | `1` | 0.5-10 |
| `feature_types` | Comparison types | `real` | See below |
| `duration_method` | DUR aggregation | `geometric` | `geometric`, `arithmetic`, `median`, `trimmed_mean` |

### Feature List Options

| Code | Feature |
|:-----|:--------|
| `COV` | Coverage |
| `OCC` | Occurrence |
| `DUR` | Mean microstate duration |
| `MMD` | Alias for `DUR` |
| `TP` | Transition Probabilities |
| `GEV` | Global Explained Variance |
| `LZC` | Lempel-Ziv Complexity |
| `ER` | Entropy Rate |
| `HE` | Hurst Exponent |
| `ERR` | Entropy Representation Ratio |
| `ROF` | Relative Occurrence Frequency (epoched data) |
| `RTF` | Relative Transition Frequency (epoched data) |

{: .note }
> `DUR` and `MMD` select the same computation, so either name may be used. Both produce `DUR_<microstate>` columns in the output, and listing both does not duplicate the feature.

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
feature_mode = averaged
feature_types = real
duration_method = geometric
```

#### Comprehensive Analysis

```ini
[features_config]
export_format = .csv
feature_list = COV,OCC,MMD,GEV,TP,LZC,ER,HE
feature_mode = averaged
feature_types = real,surrogate
duration_method = geometric
```

#### COV-DUR-OCC Algebraic Consistency

Use the arithmetic aggregation when DUR must satisfy the identity in [Relationship](#relationship) exactly (e.g. when reporting all three metrics in tables that should add up):

```ini
[features_config]
feature_list = COV,OCC,MMD
feature_mode = averaged
duration_method = arithmetic
```

#### Sliding Analysis

```ini
[features_config]
export_format = .csv
feature_list = COV,OCC,MMD,GEV
feature_mode = sliding
sliding_window_size = 2
feature_types = real
```

---

## Output Files

### CSV Format

Standard comma-separated values, one row per input file and one column per feature/microstate pair. COV and GEV are percentages, OCC is in Hz and DUR in milliseconds:

```csv
Filename,COV_A,COV_B,DUR_A,DUR_B,GEV_A,GEV_B,OCC_A,OCC_B
sub-01,28.4,24.1,87.5,82.7,22.3,18.4,3.2,2.9
...
```

### Transition Matrix

Transition probabilities share the features file, with one column per ordered pair named `TP_<from>_<to>`. Self-transitions and pairs that never occur are absent rather than zero, and the columns leaving each state sum to 1:

```csv
Filename,TP_A_B,TP_A_C,TP_A_D,TP_B_A,TP_B_C,TP_B_D
sub-01,0.35,0.40,0.25,0.30,0.45,0.25
...
```

### Sliding Output

Window-indexed features, with consecutive non-overlapping windows:

```csv
Filename,Window_index,COV_A,DUR_A,OCC_A
sub-01,0,30.2,85.7,3.5
sub-01,1,27.4,87.1,3.1
...
```

---

## Best Practices

1. **Select appropriate features for research question**
   - Basic: COV, OCC, DUR
   - Sequence analysis: TP, ER, LZC
   - Dynamics: HE, ERR

2. **Match mode to experimental design**
   - Averaged for between-subject comparisons
   - Sliding for within-session dynamics
   - Pre/post event for trial-level analysis

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

## References

- Lehmann, D., Ozaki, H., & Pal, I. (1987). EEG alpha map series: Brain micro-states by space-oriented adaptive segmentation. *Electroencephalography and Clinical Neurophysiology*, 67(3), 271–288. [https://doi.org/10.1016/0013-4694(87)90025-3](https://doi.org/10.1016/0013-4694(87)90025-3)
- Koenig, T., Prichep, L., Lehmann, D., Valdes Sosa, P., Braeker, E., Kleinlogel, H., Isenhart, R., & John, E. R. (2002). Millisecond by millisecond, year by year: Normative EEG microstates and developmental stages. *NeuroImage*, 16(1), 41–48. [https://doi.org/10.1006/nimg.2002.1070](https://doi.org/10.1006/nimg.2002.1070)
- Khanna, A., Pascual-Leone, A., Michel, C. M., & Farzan, F. (2015). Microstates in resting-state EEG: Current status and future directions. *Neuroscience & Biobehavioral Reviews*, 49, 105–113. [https://doi.org/10.1016/j.neubiorev.2014.12.010](https://doi.org/10.1016/j.neubiorev.2014.12.010)
- Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: A review. *NeuroImage*, 180, 577–593. [https://doi.org/10.1016/j.neuroimage.2017.11.062](https://doi.org/10.1016/j.neuroimage.2017.11.062)
- Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. *IEEE Transactions on Information Theory*, 22(1), 75–81. [https://doi.org/10.1109/TIT.1976.1055501](https://doi.org/10.1109/TIT.1976.1055501)
- von Wegner, F., Tagliazucchi, E., & Laufs, H. (2017). Information-theoretical analysis of resting state EEG microstate sequences — non-Markovianity, non-stationarity and periodicities. *NeuroImage*, 158, 99–111. [https://doi.org/10.1016/j.neuroimage.2017.06.062](https://doi.org/10.1016/j.neuroimage.2017.06.062)
- Van de Ville, D., Britz, J., & Michel, C. M. (2010). EEG microstate sequences in healthy humans at rest reveal scale-free dynamics. *Proceedings of the National Academy of Sciences*, 107(42), 18179–18184. [https://doi.org/10.1073/pnas.1007841107](https://doi.org/10.1073/pnas.1007841107)

---

## Next Step

[**Statistical Analysis Module →**]({% link modules/statistical-analysis.md %})

