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
> Evidence indicates that choosing fewer than 4 microstates may oversimplify network dynamics. Investigating 5 or more states is recommended for reliable results.

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

#### 4. Dunn Index

Ratio of minimum inter-cluster to maximum intra-cluster correlation.

| Interpretation | Best K |
|:---------------|:-------|
| Higher values | Compact, well-separated patterns |

#### 5. Davies-Bouldin Index

Average similarity between each cluster and its most similar counterpart.

| Interpretation | Best K |
|:---------------|:-------|
| Lower values | Better-separated configurations |

#### 6. Calinski-Harabasz Index

Ratio of between-cluster to within-cluster variance.

| Interpretation | Best K |
|:---------------|:-------|
| Higher values | Stronger coherence, clearer boundaries |

---

### Category C: Statistical Inference Methods

These methods evaluate clustering quality relative to null hypotheses.

#### 7. Gap Statistic

Compares observed clustering quality against expectations from random topographic distributions.

| Interpretation | Best K |
|:---------------|:-------|
| Maximum gap | Solutions exceeding chance-level organization |

#### 8. AIC (Akaike Information Criterion)

Model selection criterion penalizing solutions with excessive parameters.

| Interpretation | Best K |
|:---------------|:-------|
| Minimum AIC | Best balance of fit and complexity |

#### 9. BIC (Bayesian Information Criterion)

Similar to AIC but with stronger penalty for model complexity.

| Interpretation | Best K |
|:---------------|:-------|
| Minimum BIC | More parsimonious solutions |

#### 10. Krzanowski-Lai Criterion

Evaluates relative improvement across successive cluster numbers.

| Interpretation | Best K |
|:---------------|:-------|
| Maximum value | Where additional clusters provide meaningful differentiation |

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
4. In case of ties, use CV or BIC as tiebreaker

---

## Configuration

### Parameters

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `number_of_maps` | Number of clusters | `4` | Integer or `auto` |
| `kmin` | Minimum K to evaluate | `2` | 2-10 |
| `kmax` | Maximum K to evaluate | `10` | 4-15 |
| `stopping_mode` | Selection strategy | `majority_vote` | See below |
| `stopping_parameter` | Criterion threshold | `10` | 1-100 |

### Stopping Modes

| Mode | Description |
|:-----|:------------|
| `majority_vote` | Consensus across all criteria |
| `gev` | Use Global Explained Variance |
| `cv` | Use Cross-Validation Criterion |
| `sil` | Use Silhouette Score |
| `ch` | Use Calinski-Harabasz Index |
| `db` | Use Davies-Bouldin Index |
| `residual` | Use residual variance |

### Example Configuration

```ini
[clustering_config]
# Auto-select K using validation
number_of_maps = auto
kmin = 4
kmax = 8
stopping_mode = majority_vote
stopping_parameter = 10
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

## Next Step

[**Microstate Clustering Module →**]({% link modules/clustering.md %})

