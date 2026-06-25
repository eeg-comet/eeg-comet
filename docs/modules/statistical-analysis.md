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

| Test | Design | Assumption |
|:-----|:-------|:-----------|
| **Paired t-test** | Within-subject (repeated measures) | Paired observations |
| **Independent t-test** | Between-subject (groups) | Equal variances |
| **Welch's t-test** | Between-subject | Unequal variances (automatic) |

### Variance Homogeneity

Welch's t-test (Welch, 1947) is automatically applied when Levene's test indicates unequal variances, providing robust inference without requiring equal variance assumptions.

### Multiple Comparison Correction

Given comparisons across multiple microstates and features:

| Method | Control | Best For |
|:-------|:--------|:---------|
| **FDR** (False Discovery Rate) | Expected false positives | Exploratory analyses |
| **Bonferroni** | Family-wise error | Confirmatory analyses |

The FDR procedure controls the expected proportion of false positives (Benjamini & Hochberg, 1995).

**Number of comparisons:**
- K microstates × N features = total comparisons
- Example: 4 microstates × 4 features = 16 comparisons

### Example Application

```
Research question: Does meditation training affect microstate dynamics?

Design: Pre-training vs. Post-training (within-subject)
Test: Paired t-tests on COV, OCC, DUR for each microstate
Correction: FDR at q = 0.05
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

**Working Correlation Structures:**

| Structure | Assumption |
|:----------|:-----------|
| Independent | No within-subject correlation |
| Exchangeable | Equal correlation between all trials |
| Autoregressive | Correlation decays with trial distance |
| Unstructured | Estimate all pairwise correlations |

**Best for:** Exploratory analyses where precise correlation pattern is unknown.

### Framework 2: Linear Mixed-Effects Models (LMM)

**Subject-specific inference** explicitly modeling random effects (Laird & Ware, 1982).

| Feature | Description |
|:--------|:------------|
| **Approach** | Parametric, hierarchical |
| **Inference type** | Subject-specific parameters |
| **Random effects** | Subject intercepts/slopes |
| **Fixed effects** | Experimental conditions |

**Model specification:**

$$Y_{ij} = \beta_0 + \beta_1 X_{ij} + u_i + \varepsilon_{ij}$$

Where:
- $$Y_{ij}$$ = Feature for trial j in subject i
- $$\beta_0, \beta_1$$ = Fixed effects
- $$u_i$$ = Random subject effect
- $$\varepsilon_{ij}$$ = Residual error

### Framework 3: Generalized Linear Mixed Models (GLMM)

**Extension of LMM** for non-normally distributed outcomes.

| Distribution | Link Function | Use Case |
|:-------------|:--------------|:---------|
| Gamma | Log | Positive continuous (durations) |
| Inverse Gaussian | Log | Positive continuous, right-skewed |
| Poisson | Log | Count data (occurrences) |

**Best for:** Duration and occurrence metrics with characteristic right skew.

### Choosing Between Frameworks

| Factor | GEE | LMM/GLMM |
|:-------|:----|:---------|
| **Research question** | Population effects | Subject-specific effects |
| **Correlation structure** | Unknown | Can be modeled |
| **Distribution** | Flexible | Normal (LMM) or specified (GLMM) |
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

**Enhancement** that eliminates arbitrary threshold selection (Smith & Nichols, 2009).

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

### Output

- Time intervals of significant modulation
- Direction of effect (increase/decrease)
- Cluster-corrected p-values

---

## Paired Condition Comparisons

### Purpose

Assess condition-specific differences in event-related microstate dynamics.

### Method

```
1. Compute ROF difference between conditions per subject
2. Apply one-sample TFCE permutation to differences
3. Identify significant deviation from zero
```

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
| `statsmodels` | GEE, LMM/GLMM, parametric tests |
| `mne` | Cluster permutation, TFCE |
| `scipy` | Basic statistical tests |

### Configuration

```ini
[statistics_config]
test_type = paired
correction = fdr
alpha = 0.05
n_permutations = 5000
tfce = True
```

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
   - Averaged: Parametric t-tests
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

5. **Use sufficient permutations**
   - Minimum 1000 for p < 0.05
   - 5000+ recommended for p < 0.01

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

- Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika*, 34(1–2), 28–35. [https://doi.org/10.1093/biomet/34.1-2.28](https://doi.org/10.1093/biomet/34.1-2.28)
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

