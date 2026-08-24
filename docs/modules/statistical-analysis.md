---
title: 9. Statistical Analysis
layout: default
parent: Modules
nav_order: 9
description: "Statistical Analysis Module - Inference methods for microstate data"
---

# Statistical Analysis Module
{: .no_toc }

Parametric tests, regression frameworks, and permutation-based methods for microstate inference.
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

Comprehensive statistical analysis capabilities address the hierarchical data structures and multiple-comparison challenges inherent to microstate analysis. Three complementary analytical approaches are provided:

1. **Averaged Metrics** - Parametric comparisons of recording-level features
2. **Trial-Level Data** - Regression frameworks for hierarchical data
3. **Temporal Dynamics** - Permutation-based inference for time series

---

## Inference for Averaged Metrics

### Purpose

Compare microstate features calculated over entire recordings between conditions or groups.

### Available Tests

The test is determined by the **Study Design** and **Statistical Test Type** selections in the Compare Studies window:

| Test | Design | Test Type | Assumption |
|:-----|:-------|:----------|:-----------|
| **Paired t-test** | Paired (Within-Subject) | Parametric | Paired observations, normality |
| **Independent t-test** | Independent (Between-Subject) | Parametric | Normality; unequal variances handled automatically |
| **Wilcoxon signed-rank** | Paired (Within-Subject) | Non-parametric | Paired observations |
| **Mann-Whitney U** | Independent (Between-Subject) | Non-parametric | Independent samples |

The independent t-test picks its variance assumption from the data. Levene's test (median-centred) is run on the two samples first: if it returns p < 0.05 the test switches to Welch's unequal-variance formulation, otherwise the pooled-variance (Student's) formulation is used. Levene's test is only run when both groups have at least three observations, so smaller samples always fall back to Student's test, as does degenerate input such as zero variance in both groups. Paired tests require samples of equal length; if the two studies contribute different numbers of files, both are truncated to the length of the smaller one before pairing.

{: .note }
> The **Statistical Test Model** dropdown lists the model names available for each combination of options. For averaged metrics the test that is run is fixed by the Study Design and Statistical Test Type radio buttons; the selected model name is echoed in the analysis header.

### Multiple Comparison Correction

Corrections are applied with `statsmodels.stats.multitest.multipletests`. The Compare Studies window offers:

| Method | Control | Best For |
|:-------|:--------|:---------|
| **Bonferroni** | Family-wise error | Confirmatory analyses |
| **Holm** | Family-wise error | Step-down alternative to Bonferroni |
| **Sidak** | Family-wise error | Independent tests |
| **Holm-Sidak** | Family-wise error | Step-down form of Sidak |
| **Hommel** | Family-wise error | Positively dependent tests |
| **FDR-BH** | Expected false positives | Exploratory analyses |
| **FDR-TSBH** | Expected false positives | Two-stage Benjamini–Hochberg |
| **FDR-TSBKY** | Expected false positives | Two-stage Benjamini–Krieger–Yekutieli |

The FDR procedures control the expected proportion of false positives (Benjamini & Hochberg, 1995).

**Number of comparisons:**
- Features are analysed one at a time, so correction spans the microstate columns of the currently selected feature
- Example: 4 comparisons for a 4-class solution

### Example Application

```
Research question: Does meditation training affect microstate dynamics?

Design: Pre-training vs. Post-training (within-subject)
Test: Paired t-tests on COV, OCC and DUR, one feature at a time
Correction: FDR-BH at q = 0.05
```

---

## Inference for Trial-Level Data

### The Challenge

Comparing microstate features before and after events at the single-trial level presents statistical challenges:

1. **Non-normal distributions** - Microstate durations often show positive skew
2. **Within-subject correlations** - Trials within subjects are not independent
3. **Hierarchical structure** - Trials nested within subjects

{: .warning }
> Ignoring within-subject correlations inflates Type I error rates and produces unreliable inferences.

### Framework 1: Generalized Estimating Equations (GEE)

**Population-averaged inference** accounting for within-subject correlations (Liang & Zeger, 1986).

| Feature | Description |
|:--------|:------------|
| **Approach** | Semi-parametric, robust |
| **Inference type** | Population-averaged effects |
| **Correlation handling** | Working correlation structure |
| **Robustness** | Valid under correlation misspecification |

**Model specification:** `feature ~ Condition`, grouped by subject. The distribution family follows the selected model entry: **Generalized Estimating Equations (GEE, Gaussian/identity)**, offered under the parametric test type, uses a Gaussian family with an identity link, while **Generalized Estimating Equations (GEE, Gamma/log)**, offered under the non-parametric test type, uses a Gamma family with a log link. Subject identifiers are derived from the filenames, and at least three subjects are required.

**Working Correlation Structures:**

| Structure | Assumption | Applied when |
|:----------|:-----------|:-------------|
| Exchangeable | Equal correlation between all trials | Study Design = Paired (Within-Subject) |
| Independent | No within-subject correlation | Study Design = Independent (Between-Subject) |

**Best for:** Exploratory analyses where precise correlation pattern is unknown.

### Framework 2: Linear Mixed-Effects Models (LMM)

**Subject-specific inference** explicitly modeling random effects (Laird & Ware, 1982).

| Feature | Description |
|:--------|:------------|
| **Approach** | Parametric, hierarchical |
| **Inference type** | Subject-specific parameters |
| **Random effects** | Random subject intercept |
| **Fixed effects** | Experimental condition |

**Model specification:**

$$Y_{ij} = \beta_0 + \beta_1 X_{ij} + u_i + \varepsilon_{ij}$$

Where:
- $$Y_{ij}$$ = Feature for trial j in subject i
- $$\beta_0, \beta_1$$ = Fixed effects
- $$u_i$$ = Random subject effect
- $$\varepsilon_{ij}$$ = Residual error

The mixed model `Value ~ Group` with a random subject intercept is fitted with `statsmodels` for the event-related (pre/post window) analysis and for sliding-window comparisons between studies. Effect sizes are reported as Cohen's d computed from the fixed-effect coefficient and the residual scale, together with 95% confidence intervals and a convergence flag.

{: .note }
> In the trial-level branch, both **Linear Mixed Model (LMM)** and **Generalized Estimating Equations (GEE, Gaussian/identity)** fit the Gaussian GEE model described above.

### Framework 3: Gamma Family for Skewed Outcomes

**Extension of the GEE model** for non-normally distributed outcomes such as durations and coverage with characteristic right skew.

Selecting **Generalized Estimating Equations (GEE, Gamma/log)** fits the population-averaged model of Framework 1 with a Gamma family and a log link, which suits strictly positive, right-skewed outcomes such as trial-level coverage. The log link requires a strictly positive response, so if the feature contains values at or below zero the whole feature is shifted upwards before fitting. That shift is applied only on the Gamma/log path; under an identity link it would bias the intercept.

The remaining non-parametric trial-level entries — **Permutation Test with Clustering** for a paired design and **Bootstrap Resampling** for an independent one — fall back to the subject-level analysis described above, and the results panel states that the aggregation was used.

### Choosing Between Frameworks

| Factor | GEE | LMM |
|:-------|:----|:----|
| **Research question** | Population effects | Subject-specific effects |
| **Correlation structure** | Exchangeable or independent working structure | Random subject intercept |
| **Distribution** | Gaussian with identity link, or Gamma with log link | Normal residuals |
| **Complexity** | Simpler | More detailed |

---

## Inference for Temporal Dynamics

### The Challenge

Event-related analyses with hundreds of timepoints create:
- Massive multiple comparison problem
- Temporal autocorrelation between adjacent points
- Need to detect both transient and sustained effects

### Cluster-Based Permutation Testing

**Principle:** Test clusters of adjacent timepoints showing consistent effects, rather than individual samples (Maris & Oostenveld, 2007).

**Algorithm:**

```
1. Compute test statistic at each timepoint
2. Identify clusters of adjacent points exceeding threshold
3. Sum statistics within each cluster (cluster mass)
4. Generate null distribution via permutation:
   - Randomly reassign condition labels
   - Repeat clustering procedure
   - Record maximum cluster mass
5. Compare observed clusters to null distribution
6. Report clusters with p < 0.05 (corrected)
```

### Threshold-Free Cluster Enhancement (TFCE)

**Enhancement** that eliminates arbitrary threshold selection (Smith & Nichols, 2009). EEG-COMET applies the cluster-permutation framework in this threshold-free form: ROF time-course tests always use TFCE rather than a fixed cluster-forming threshold.

$$\text{TFCE}(t) = \int_0^{h(t)} e(h)^E \cdot h^H \, dh$$

Where:
- $$h(t)$$ = statistic height at point t
- $$e(h)$$ = cluster extent at height h
- $$E, H$$ = enhancement parameters

| Advantage | Description |
|:----------|:------------|
| No threshold | Integrates across all thresholds |
| Sensitivity | Comparable to cluster methods |
| Spatial specificity | Better localization |

---

## Single-Condition Event Effects

### Purpose

Test whether experimental events significantly modulate microstate dynamics within a single condition.

### Method: Sign-Flipping Permutation

```
For each permutation (5000 iterations):
    1. Randomly flip sign of each subject's ROF time course
    2. Compute one-sample t-statistics
    3. Apply TFCE transformation
    4. Record maximum TFCE value

Compare observed TFCE to null distribution
Report significant clusters at p < 0.05
```

The test is restricted to the 20–1000 ms post-event window, and operates on ROF values that have already been baseline-corrected against the pre-event median and CLR-transformed. At least three subjects are required.

### Output

- Time intervals of significant modulation
- Direction of effect (increase/decrease)
- Cluster-corrected p-values
- Cohen's d per significant cluster

---

## Paired Condition Comparisons

### Purpose

Assess condition-specific differences in event-related microstate dynamics.

### Method

```
1. Match subjects across the two studies by subject identifier
2. Compute ROF difference between conditions per matched subject
3. Apply one-sample TFCE permutation to differences
4. Identify significant deviation from zero
```

At least three matched subjects are required per microstate. This comparison is available for the paired design; an independent-samples ROF comparison is not currently offered, and selecting it reports that the paired design should be used instead.

### Multiple Comparisons

When comparing multiple condition pairs:
- Apply Bonferroni across contrasts
- Or use hypothesis-driven subset

---

## Transition Frequency Analysis

For **Relative Transition Frequency (RTF)** analyses:

| Challenge | Solution |
|:----------|:---------|
| Large hypothesis space | Focus on specific transitions |
| Many transition pairs | Bonferroni correction |
| Theory-driven selection | Reduce multiple comparisons |

---

## Implementation Details

### Software Foundation

EEG-COMET uses established Python packages:

| Package | Purpose |
|:--------|:--------|
| `statsmodels` | GEE, mixed-effects models, multiple-comparison correction |
| `mne` | Cluster permutation, TFCE |
| `scipy` | t-tests, Levene's test, Wilcoxon signed-rank, Mann-Whitney U |

### Configuration

Statistical analysis is configured entirely through the Compare Studies window; it has no section in the study configuration file. The relevant controls are:

| Control | Effect |
|:--------|:-------|
| **Analysis Data** | Full Recording or Event-Related (Pre/Post) |
| **Analysis Level** | Subject-Level or Trial-Level |
| **Study Design** | Paired (Within-Subject) or Independent (Between-Subject) |
| **Statistical Test Type** | Parametric or Non-parametric |
| **Statistical Test Model** | Model names available for the current combination |
| **Multiple Comparison Correction Method** | Correction applied to the p-values |

Permutation settings are not exposed. ROF cluster tests always run 5,000 sign-flipping permutations with TFCE parameters `start = 0` and `step = 0.2`, two-tailed, and report clusters at p < 0.05.

When a ROF feature is selected, the analysis options are disabled and the model is fixed to TFCE.

---

## Reporting Guidelines

### For Averaged Metrics

Report:
- Test statistic (t-value)
- Degrees of freedom
- Uncorrected p-value
- Corrected p-value and method
- Effect size (Cohen's d)

### For Trial-Level Models

Report:
- Model specification (fixed/random effects)
- Coefficient estimates and standard errors
- Confidence intervals
- Model fit statistics (AIC, BIC)

### For Permutation Tests

Report:
- Number of permutations
- Cluster-forming threshold (if applicable)
- TFCE parameters (if used)
- Significant cluster extent (time range)
- Cluster p-value

---

## Best Practices

1. **Match test to data structure**
   - Averaged: t-tests or rank-based tests
   - Trial-level: GEE or LMM
   - Temporal: Permutation

2. **Always correct for multiple comparisons**
   - FDR for exploration
   - Bonferroni for confirmation

3. **Check assumptions**
   - Normality for parametric tests
   - Correlation structure for GEE
   - Random effects distribution for LMM

4. **Report effect sizes**
   - Statistical significance ≠ practical importance

5. **Note the fixed permutation count**
   - ROF cluster tests use 5,000 permutations
   - Sufficient for the p < 0.05 cluster threshold that is applied

---

## Troubleshooting

<div class="callout warning">
<strong>GEE fails to converge</strong><br>
Try different working correlation structure. Scale continuous predictors. Check for collinearity.
</div>

<div class="callout warning">
<strong>LMM singular fit</strong><br>
Random effects variance estimated at zero. Simplify random effects structure. More data may be needed.
</div>

<div class="callout warning">
<strong>No significant clusters</strong><br>
Effect may be small or variable. Increase sample size. Consider ROI-based approach.
</div>

---

## References

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. [https://doi.org/10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- Liang, K.-Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. *Biometrika*, 73(1), 13–22. [https://doi.org/10.1093/biomet/73.1.13](https://doi.org/10.1093/biomet/73.1.13)
- Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. *Biometrics*, 38(4), 963–974. [https://doi.org/10.2307/2529876](https://doi.org/10.2307/2529876)
- Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data. *Journal of Neuroscience Methods*, 164(1), 177–190. [https://doi.org/10.1016/j.jneumeth.2007.03.024](https://doi.org/10.1016/j.jneumeth.2007.03.024)
- Smith, S. M., & Nichols, T. E. (2009). Threshold-free cluster enhancement: Addressing problems of smoothing, threshold dependence and localisation in cluster inference. *NeuroImage*, 44(1), 83–98. [https://doi.org/10.1016/j.neuroimage.2008.03.061](https://doi.org/10.1016/j.neuroimage.2008.03.061)
- Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. [https://doi.org/10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*, 92–96. [https://doi.org/10.25080/Majora-92bf1922-011](https://doi.org/10.25080/Majora-92bf1922-011)

---

## Next Step

[**Source Localization Module →**]({% link modules/source-localization.md %})

