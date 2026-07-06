---
title: 4. Cluster Validation
layout: default
parent: Modules
nav_order: 4
description: "Cluster Number Validation Module - Determining optimal microstate count"
---

# Cluster Number Validation Module
{: .no_toc }

Comprehensive validation framework with 10 statistical methods and 3 selection strategies.
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

Selecting the optimal number of microstate templates is an important methodological choice that directly impacts analysis results and interpretation. EEG-COMET provides a comprehensive validation framework combining:

- **10 complementary statistical methods**
- **3 selection strategies**

All validation methods use spatial correlation as the primary similarity measure, treating topographies differing only in voltage sign as the same underlying neural configuration (polarity invariance).

{: .note }
> Evidence indicates that choosing fewer than 4 microstates may oversimplify network dynamics. Investigating 5 or more states is recommended for reliable results (Michel & Koenig, 2018; Koenig et al., 2024).

### How the criteria are computed

All ten criteria are evaluated on a polarity-invariant basis: cluster membership and every distance/dispersion term are derived from the absolute spatial correlation between a topography and its assigned template, so that maps differing only in voltage sign are treated as the same configuration (Pascual-Marqui et al., 1995; Michel & Koenig, 2018). For each candidate K in `[k_min, k_max]`, clustering is run once and all selected criteria are scored from that solution. Within-cluster dispersion uses the polarity-invariant distance `d = 1 - |r|` (Tibshirani et al., 2001), and the information criteria treat each template as an average-referenced, unit-normalized topography with `C - 1` free parameters (so the model has `K x (C - 1)` parameters in total).

---

## Validation Methods

### Category A: Variance-Based Methods

These methods quantify how efficiently different cluster numbers capture topographic information.

#### 1. Cross-Validation Criterion (CV)

Penalizes excessive model complexity through variance decomposition.

| Interpretation | Best K |
|:---------------|:-------|
| Lower values | Better balance of explanatory power and parsimony |

**Reference:** Pascual-Marqui et al., 1995

#### 2. Global Explained Variance (GEV)

Measures the proportion of total topographic variance explained by templates.

| Interpretation | Best K |
|:---------------|:-------|
| Look for "elbow" | Where additional clusters yield diminishing returns |

**Reference:** Pascual-Marqui et al., 1995; Michel & Koenig, 2018

---

### Category B: Separation-Based Methods

These methods quantify distinctiveness and compactness of identified clusters.

#### 3. Silhouette Analysis

Measures clustering consistency by comparing within-cluster similarity to between-cluster separation.

| Score Range | Interpretation |
|:------------|:---------------|
| Close to +1 | Well-matched assignments, clear distinctions |
| Close to 0 | Overlapping clusters |
| Negative | Possible misassignments |

**Reference:** Rousseeuw, 1987

#### 4. Dunn Index

Ratio of minimum inter-cluster to maximum intra-cluster correlation.

| Interpretation | Best K |
|:---------------|:-------|
| Higher values | Compact, well-separated patterns |

**Reference:** Dunn, 1974

#### 5. Davies-Bouldin Index

Average similarity between each cluster and its most similar counterpart.

| Interpretation | Best K |
|:---------------|:-------|
| Lower values | Better-separated configurations |

**Reference:** Davies & Bouldin, 1979

#### 6. Calinski-Harabasz Index

Ratio of between-cluster to within-cluster variance.

| Interpretation | Best K |
|:---------------|:-------|
| Higher values | Stronger coherence, clearer boundaries |

**Reference:** Caliński & Harabasz, 1974

---

### Category C: Statistical Inference Methods

These methods evaluate clustering quality relative to null hypotheses.

#### 7. Gap Statistic

Compares observed clustering quality against expectations from random topographic distributions.

| Interpretation | Best K |
|:---------------|:-------|
| Maximum gap | Solutions exceeding chance-level organization |

**Reference:** Tibshirani, Walther & Hastie, 2001

#### 8. AIC (Akaike Information Criterion)

Model selection criterion penalizing solutions with excessive parameters.

| Interpretation | Best K |
|:---------------|:-------|
| Minimum AIC | Best balance of fit and complexity |

**Reference:** Akaike, 1974

#### 9. BIC (Bayesian Information Criterion)

Similar to AIC but with stronger penalty for model complexity.

| Interpretation | Best K |
|:---------------|:-------|
| Minimum BIC | More parsimonious solutions |

**Reference:** Schwarz, 1978

#### 10. Krzanowski-Lai Criterion

Evaluates relative improvement across successive cluster numbers.

| Interpretation | Best K |
|:---------------|:-------|
| Maximum value | Where additional clusters provide meaningful differentiation |

**Reference:** Krzanowski & Lai, 1988

---

## Method Summary Table

| Method | Category | Optimal | Principle |
|:-------|:---------|:--------|:----------|
| Cross-Validation | Variance | Minimum | Variance decomposition |
| GEV | Variance | Elbow | Explained variance |
| Silhouette | Separation | Maximum | Cluster consistency |
| Dunn | Separation | Maximum | Separation/compactness ratio |
| Davies-Bouldin | Separation | Minimum | Inter-cluster similarity |
| Calinski-Harabasz | Separation | Maximum | Variance ratio |
| Gap Statistic | Inference | Maximum | vs. random distribution |
| AIC | Inference | Minimum | Fit vs. complexity |
| BIC | Inference | Minimum | Fit vs. complexity (stricter) |
| Krzanowski-Lai | Inference | Maximum | Relative improvement |

---

## Selection Strategies

### Strategy 1: Manual Inspection

View validation curves and decide based on domain knowledge.

**When to use:**
- Domain expertise helps interpret patterns
- Theoretical considerations suggest specific K values
- Disagreements between criteria need investigation

**Process:**
1. Plot all validation metrics against K values
2. Identify convergence patterns
3. Note systematic disagreements
4. Consider theoretical constraints
5. Select K with supporting evidence

### Strategy 2: Automatic Selection

Use a single criterion to determine K automatically.

**When to use:**
- Specific criterion is most relevant to research question
- Streamlined, reproducible workflow needed
- Theoretical reasons favor particular metric

**Common choices:**
- **CV** - Traditional, conservative
- **GEV** - Maximize explained variance
- **Silhouette** - Maximize cluster quality

### Strategy 3: Majority Vote

Select K most frequently chosen across all criteria.

**When to use:**
- No strong theoretical preference
- Want robust, multi-perspective decision
- Exploratory analysis

**Process:**
1. Compute optimal K for each criterion
2. Tally votes for each K value
3. Select K with most votes
4. In case of ties, the smallest K among the tied values is chosen (the more parsimonious solution)

{: .note }
> Combining multiple complementary criteria rather than relying on a single index is recommended for objective, reproducible microstate-count selection (Michel & Koenig, 2018; Koenig et al., 2024; Michel et al., 2024).

---

## Configuration

### Parameters

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `n_maps` | Number of clusters | `4` | Integer or `auto` |
| `k_min` | Minimum K to evaluate | `2` | 2-10 |
| `k_max` | Maximum K to evaluate | `10` | 4-15 |
| `stopping_mode` | Selection strategy | `majority_vote` | See below |
| `stopping_threshold` | GEV elbow gain threshold (percent): the minimum relative GEV increase that justifies an additional cluster | `10` | 1-100 |

### Stopping Modes

| Mode | Description |
|:-----|:------------|
| `majority_vote` | Consensus across all criteria |
| `gev` | Use Global Explained Variance |
| `cv` | Use Cross-Validation Criterion |
| `db` | Use Davies-Bouldin Index |
| `kl` | Use Krzanowski-Lai Criterion |
| `sil` | Use Silhouette Score |
| `dunn` | Use Dunn Index |
| `ch` | Use Calinski-Harabasz Index |
| `gap` | Use Gap Statistic |
| `aic` | Use Akaike Information Criterion |
| `bic` | Use Bayesian Information Criterion |

### Example Configuration

```ini
[clustering_config]
# Auto-select K using validation
n_maps = auto
k_min = 4
k_max = 8
stopping_mode = majority_vote
stopping_threshold = 10
```

---

## Interpreting Results

### Ideal Patterns

| Pattern | Interpretation |
|:--------|:---------------|
| All methods agree | Strong evidence for optimal K |
| Most methods converge | Robust choice |
| Methods disagree | Investigate specific patterns |

### Common Disagreements

| Scenario | Explanation | Resolution |
|:---------|:------------|:-----------|
| Silhouette peaks early, GEV late | Trade-off between separation and coverage | Consider research goals |
| BIC favors fewer than AIC | Stricter complexity penalty | Use BIC for parsimony |
| Gap statistic differs | Comparison to random may not fit data | Weight other methods |

### Red Flags

{: .warning }
> - All methods favor K=2 (data may lack clear structure)
> - No clear pattern across methods (consider data quality)
> - Extreme values (K=10+) optimal (possible overfitting)

---

## Visualization

EEG-COMET provides visualization tools for validation results:

### Validation Curves

- Plot each criterion against K
- Identify elbow points
- Compare method recommendations

### Voting Summary

- Bar chart of votes per K
- Highlights consensus
- Shows disagreement distribution

### Template Preview

- View templates at different K values
- Compare interpretability
- Assess topographic distinctiveness

---

## Best Practices

1. **Evaluate a reasonable range**
   - Start with K = 4-8 for typical analyses
   - Extend range if edge values are selected

2. **Consider theoretical constraints**
   - Prior studies often find 4-7 canonical states
   - Clinical populations may differ

3. **Use multiple criteria**
   - Don't rely on single method
   - Majority vote provides robustness

4. **Inspect templates at chosen K**
   - Are topographies interpretable?
   - Do they match expected patterns?

5. **Document the selection process**
   - Record all validation results
   - Note any manual adjustments

---

## Computational Considerations

### Time Complexity

Validation requires running clustering at each K value:

| K Range | Approximate Time |
|:--------|:-----------------|
| K = 4-6 | Fast (minutes) |
| K = 2-10 | Moderate (10-30 min) |
| K = 2-15 | Longer (30+ min) |

### Memory Usage

Gap statistic requires generating null distributions, increasing memory needs.

### Optimization Tips

- Use GFP peaks for initial validation
- Run detailed validation on representative subset
- Parallelize across K values when possible

---

## References

- Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of brain electrical activity into microstates: model estimation and validation. *IEEE Transactions on Biomedical Engineering*, 42(7), 658–665. [https://doi.org/10.1109/10.391164](https://doi.org/10.1109/10.391164)
- Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: A review. *NeuroImage*, 180, 577–593. [https://doi.org/10.1016/j.neuroimage.2017.11.062](https://doi.org/10.1016/j.neuroimage.2017.11.062)
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. [https://doi.org/10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)
- Dunn, J. C. (1974). Well-separated clusters and optimal fuzzy partitions. *Journal of Cybernetics*, 4(1), 95–104. [https://doi.org/10.1080/01969727408546059](https://doi.org/10.1080/01969727408546059)
- Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224–227. [https://doi.org/10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909)
- Caliński, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics*, 3(1), 1–27. [https://doi.org/10.1080/03610927408827101](https://doi.org/10.1080/03610927408827101)
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. *Journal of the Royal Statistical Society: Series B*, 63(2), 411–423. [https://doi.org/10.1111/1467-9868.00293](https://doi.org/10.1111/1467-9868.00293)
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723. [https://doi.org/10.1109/TAC.1974.1100705](https://doi.org/10.1109/TAC.1974.1100705)
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464. [https://doi.org/10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136)
- Krzanowski, W. J., & Lai, Y. T. (1988). A criterion for determining the number of groups in a data set using sum-of-squares clustering. *Biometrics*, 44(1), 23–34. [https://doi.org/10.2307/2531893](https://doi.org/10.2307/2531893)
- Koenig, T., Diezig, S., Kalburgi, S. N., et al. (2024). EEG-Meta-Microstates: Towards a more objective use of resting-state EEG microstate findings across studies. *Brain Topography*, 37(2), 218–231. [https://doi.org/10.1007/s10548-023-00993-6](https://doi.org/10.1007/s10548-023-00993-6)
- Michel, C. M., Brechet, L., Schiller, B., et al. (2024). Current state of EEG/ERP microstate research. *Brain Topography*, 37, 169–180. [https://doi.org/10.1007/s10548-024-01037-3](https://doi.org/10.1007/s10548-024-01037-3)

---

## Next Step

[**Microstate Clustering Module →**]({% link modules/clustering.md %})

