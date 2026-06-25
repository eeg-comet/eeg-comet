---
title: 5. Clustering
layout: default
parent: Modules
nav_order: 5
description: "Microstate Clustering Module - Template identification algorithms"
---

# Microstate Clustering Module
{: .no_toc }

Modified K-means and TAAHC algorithms for microstate template identification.
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

Two clustering algorithms are available for microstate template identification:

| Algorithm | Approach | Speed | Exploration |
|:----------|:---------|:------|:------------|
| **Modified K-means** | Iterative refinement | Fast | Limited by initialization |
| **TAAHC** | Top-down hierarchical | Slower | Thorough |

Both algorithms maintain polarity invariance, treating sign-inverted topographies as identical (essential because EEG reference schemes are arbitrary).

---

## Modified K-means Algorithm

### Overview

Identifies microstate templates through iterative refinement until stable solutions emerge.

### Algorithm Steps

```
1. INITIALIZE: Randomly select K topographies as initial templates
2. REPEAT:
   a. ASSIGNMENT: Match each EEG topography to template with 
      highest spatial correlation (polarity-invariant)
   b. UPDATE: Compute new templates via eigenvalue decomposition 
      of covariance matrix for each cluster
   c. CHECK: Assess convergence (GEV stability or template correlation)
3. UNTIL: Convergence or maximum iterations reached
```

### Polarity Invariance

During assignment, both polarities are tested:

$$\text{similarity} = \max(|r|, |-r|)$$

where $$r$$ is the Pearson correlation between topographies.

### Convergence Criteria

| Criterion | Description |
|:----------|:------------|
| **GEV stability** | Change in explained variance < tolerance |
| **Template stability** | Correlation between successive templates > threshold |
| **Maximum iterations** | Hard stop after N iterations |

### Advantages

- Computationally efficient
- Well-understood behavior
- Good for most applications

### Limitations

- Can get trapped in local minima
- Sensitive to initialization
- Multiple runs recommended

---

## TAAHC Algorithm

### Overview

**Topographic Atomize and Agglomerate Hierarchical Clustering** uses a top-down approach, starting with maximum fragmentation and gradually consolidating clusters.

### Algorithm Steps

```
1. INITIALIZE: Treat each timepoint as its own cluster
2. REPEAT:
   a. IDENTIFY: Find cluster with lowest internal coherence
   b. ATOMIZE: Disband this cluster
   c. REASSIGN: Assign members to remaining clusters 
      based on best spatial correlation
3. UNTIL: Desired number of clusters reached
```

### Internal Coherence

For each cluster, coherence is measured as the average spatial correlation among members:

$$\text{coherence}_k = \frac{1}{n_k(n_k-1)} \sum_{i \neq j} |r_{ij}|$$

The cluster with lowest coherence is atomized.

### Advantages

| Benefit | Description |
|:--------|:------------|
| **Escapes local minima** | Continuous reassignment corrects poor assignments |
| **No initialization sensitivity** | Deterministic from data |
| **Thorough exploration** | Considers all possible reassignments |

### Limitations

| Limitation | Mitigation |
|:-----------|:-----------|
| **Computational cost** | Mini-batch processing |
| **Memory intensive** | Apply to representative subsets |
| **Slower** | Use when thoroughness matters |

### Mini-batch Processing

For large datasets, TAAHC can be applied to representative subsets:

1. Randomly sample N timepoints
2. Apply TAAHC to sample
3. Evaluate templates on full dataset

---

## Algorithm Comparison

| Aspect | Modified K-means | TAAHC |
|:-------|:-----------------|:------|
| **Speed** | Fast | Slow |
| **Initialization** | Random (sensitive) | Deterministic |
| **Local minima** | Can get trapped | Better at escaping |
| **Parallelizable** | Yes (multiple runs) | Limited |
| **Memory** | Low | Higher |
| **Best for** | Large datasets, initial exploration | Final solution, small datasets |

---

## Configuration Parameters

### Common Parameters

| Parameter | Description | Default | Range |
|:----------|:------------|:--------|:------|
| `number_of_maps` | Number of clusters | `4` | 2-10+ |
| `max_iterations` | Iteration limit | `500` | 100-1000 |
| `clustering_tolerance` | Convergence threshold | `1e-6` | 1e-8 to 1e-4 |
| `number_of_repeats` | Number of runs | `5` | 1-100 |

### K-means Specific

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `initializer` | Initialization method | `Random` | `Random`, `K-Means++` |
| `clustering_method` | Algorithm variant | `Modified K-Means Clustering` | See below |

### Initialization Methods

| Method | Description |
|:-------|:------------|
| `Random` | Random selection of initial templates |
| `K-Means++` | Smart initialization for better spread |

K-Means++ reduces sensitivity to initialization by selecting initial centroids that are well-separated.

### Clustering Methods

| Method | Description |
|:-------|:------------|
| `Modified K-Means Clustering` | Standard polarity-invariant K-means |
| `K-Means Clustering` | Standard K-means (not polarity-invariant) |
| `PCA + K-Means Clustering` | Dimensionality reduction first |
| `Agglomerative Hierarchical Clustering` | Bottom-up hierarchical (TAAHC) |

---

## Multiple Runs

### Why Multiple Runs?

K-means results depend on initialization. Multiple runs increase chance of finding global optimum.

### Recommended Settings

| Scenario | Repeats | Rationale |
|:---------|:--------|:----------|
| Quick exploration | 5 | Fast, approximate |
| Standard analysis | 10-20 | Good balance |
| Publication | 50-100 | Robust results |
| Final validation | 100+ | Maximum confidence |

### Selection Criteria

The best run is selected based on:

1. **Global Explained Variance (GEV)** - Higher is better
2. **Template stability** - Consistent across runs

---

## Configuration Examples

### Standard Analysis

```ini
[clustering_config]
number_of_maps = 4
initializer = K-Means++
clustering_method = Modified K-Means Clustering
max_iterations = 500
clustering_tolerance = 1e-6
number_of_repeats = 20
```

### Thorough Exploration

```ini
[clustering_config]
number_of_maps = 5
initializer = Random
clustering_method = Modified K-Means Clustering
max_iterations = 1000
clustering_tolerance = 1e-8
number_of_repeats = 100
```

### TAAHC for Final Solution

```ini
[clustering_config]
number_of_maps = 4
clustering_method = Agglomerative Hierarchical Clustering
max_iterations = 500
clustering_tolerance = 1e-6
number_of_repeats = 1  # TAAHC is deterministic
```

---

## Output

### Template Topographies

For each microstate class, the algorithm outputs:

- **Template map** - The representative topography
- **Explained variance** - Proportion of variance captured
- **Cluster size** - Number of assigned timepoints

### Quality Metrics

| Metric | Description |
|:-------|:------------|
| **GEV** | Total explained variance (sum across templates) |
| **Individual GEV** | Per-template explained variance |
| **Correlation matrix** | Between-template similarities |

---

## Visualization

EEG-COMET provides visualization of clustering results:

### Template Maps

- Topographic plots for each microstate
- Consistent colormap for comparison
- Polarity shown (both orientations equivalent)

### Cluster Statistics

- Size distribution across clusters
- GEV per template
- Assignment confidence histogram

### Convergence Plots

- GEV vs. iteration
- Template stability over iterations

---

## Best Practices

1. **Start with K-Means++**
   - Better initialization reduces sensitivity

2. **Use multiple runs**
   - 20+ runs for reliable results

3. **Consider TAAHC for final solution**
   - After K-means identifies approximate K
   - More thorough exploration

4. **Check template stability**
   - Similar templates across runs indicate robustness

5. **Compare algorithms**
   - Run both K-means and TAAHC
   - Results should be similar if stable

6. **Examine correlation matrix**
   - Low inter-template correlation expected
   - High correlation suggests redundant states

---

## Troubleshooting

<div class="callout warning">
<strong>Inconsistent templates across runs</strong><br>
Increase number of repeats. Consider TAAHC for more stable solution. Check if K is appropriate.
</div>

<div class="callout warning">
<strong>Very high inter-template correlation</strong><br>
May have too many clusters. Reduce K and re-run validation.
</div>

<div class="callout warning">
<strong>Low global explained variance</strong><br>
Data may need preprocessing. Check for artifacts. Consider different frequency band.
</div>

<div class="callout warning">
<strong>Clustering fails to converge</strong><br>
Increase max_iterations. Relax tolerance. Check data quality.
</div>

---

## Next Step

[**Microstate Labeling Module →**]({% link modules/labeling.md %})

