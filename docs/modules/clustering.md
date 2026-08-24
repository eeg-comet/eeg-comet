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

Three clustering methods are available for microstate template identification:

| Algorithm | Approach | Speed | Exploration |
|:----------|:---------|:------|:------------|
| **Modified K-means** | Iterative refinement on the raw dot product | Fast | Limited by initialization |
| **Modified K-means with Spatial Similarity** | Iterative refinement on a similarity metric | Fast | Limited by initialization |
| **TAAHC** | Top-down hierarchical | Slower | Thorough |

All three methods maintain polarity invariance, treating sign-inverted topographies as identical (essential because EEG reference schemes are arbitrary).

---

## Modified K-means Algorithm

### Overview

Identifies microstate templates through iterative refinement until stable solutions emerge.

### Algorithm Steps

```
1. INITIALIZE: Select K topographies as initial templates
   (Random or K-Means++)
2. REPEAT:
   a. ASSIGNMENT: Match each EEG topography to the template with
      the highest absolute activation (polarity-invariant)
   b. UPDATE: Recompute each template as the activation-weighted
      sum of its members, then normalize to unit length
   c. CHECK: Assess convergence (improvement in residual variance)
3. UNTIL: Convergence or maximum iterations reached
```

### Polarity Invariance

Assignment ignores the sign of the match:

$$\text{similarity} = |r|$$

where $$r$$ is the correlation (dot product of normalized vectors) between topographies.

Because assignment is polarity-invariant, the template update weights every member by its **signed** activation. This sign-aligns members before summing, so that opposite-polarity topographies reinforce the template instead of cancelling out. The update is therefore not a plain arithmetic mean of the cluster members.

### Convergence Criteria

| Criterion | Description |
|:----------|:------------|
| **Residual improvement** | Improvement in residual variance < `clustering_tolerance` × residual |
| **Non-improving update** | An update that fails to reduce the residual stops the run |
| **Maximum iterations** | Hard stop after `max_iterations` |

The maps with the lowest residual seen across all iterations are returned, which are not necessarily the maps from the final iteration.

### Advantages

- Computationally efficient
- Well-understood behavior
- Good for most applications

### Limitations

- Can get trapped in local minima
- Sensitive to initialization
- Multiple runs recommended

---

## Modified K-means with Spatial Similarity

### Overview

Identical in structure to Modified K-means, but assignment and the residual are computed from an explicit similarity metric rather than the raw dot product. The metric is selected with `similarity_metric`.

| Metric | Description |
|:-------|:------------|
| `Spatial Correlation` | Pearson correlation across channels (mean-centered) |
| `Cosine Similarity` | Cosine of the angle between topographies (not mean-centered) |

### Algorithm Steps

```
1. INITIALIZE: Select K topographies as initial templates
   (Random or K-Means++)
2. REPEAT:
   a. ASSIGNMENT: Match each EEG topography to the template with
      the highest absolute similarity |s|
   b. UPDATE: Recompute each template as the sum of its members
      weighted by their signed similarity, then normalize
   c. CHECK: Assess convergence (improvement in residual)
3. UNTIL: Convergence or maximum iterations reached
```

The residual is $$1 - \overline{|s|}$$, the complement of the mean best similarity across samples. Convergence uses the same criteria as Modified K-means, and the best maps seen across iterations are returned.

**Best for:** Data where mean-centered correlation (rather than the raw inner product) is the more meaningful notion of topographic similarity.

---

## TAAHC Algorithm

### Overview

**Topographic Atomize and Agglomerate Hierarchical Clustering** uses a top-down approach, starting with maximum fragmentation and gradually consolidating clusters.

### Algorithm Steps

```
1. INITIALIZE: Detect GFP peaks and treat each peak topography
   as its own cluster
2. REPEAT:
   a. ASSIGN: Match every sample to its best-fitting cluster
      (highest absolute similarity)
   b. IDENTIFY: Find the cluster with the lowest atomization value
   c. ATOMIZE: Disband this cluster
   d. REASSIGN: Assign its members to the remaining clusters
      based on best absolute spatial similarity
   e. RECOMPUTE: Update each affected cluster centre as the first
      principal component of its members
3. UNTIL: Desired number of clusters reached
```

If fewer GFP peaks are found than the requested number of states, random timepoints are added so that clustering can start.

### Atomization Criterion

For each cluster, the atomization value sums the squared similarities of the samples assigned to it:

$$\text{value}_k = \sum_{t \in C_k} r_{t,k}^2$$

where $$r_{t,k}$$ is the polarity-invariant similarity between sample $$t$$ and its assigned template. The cluster with the **lowest** value is atomized, so both weakly-matching and sparsely-populated clusters are dissolved first (an empty cluster scores 0 and is removed immediately).

Cluster centres are recomputed as the first principal component of the member topographies rather than as an average.

### Advantages

| Benefit | Description |
|:--------|:------------|
| **Escapes local minima** | Continuous reassignment corrects poor assignments |
| **No initialization sensitivity** | Deterministic: seeded from the GFP peaks of the data |
| **Thorough exploration** | Every member of an atomized cluster is re-evaluated against all remaining templates |

### Limitations

| Limitation | Mitigation |
|:-----------|:-----------|
| **Computational cost** | Batch processing |
| **Memory intensive** | Reduce `data_percentage` to cluster on a subset |
| **Slower** | Use when thoroughness matters |

### Batch Processing

TAAHC evaluates samples in batches to bound peak memory use. The batch size is chosen automatically from the size of the data:

| Number of samples | Default batch size |
|:------------------|:-------------------|
| > 100,000 | 10,000 |
| > 50,000 | 5,000 |
| Otherwise | min(1,000, n_samples) |

Batching affects memory and speed only; the resulting templates are unchanged.

---

## Algorithm Comparison

| Aspect | Modified K-means | TAAHC |
|:-------|:-----------------|:------|
| **Speed** | Fast | Slow |
| **Initialization** | Random or K-Means++ (sensitive) | Deterministic (GFP peaks) |
| **Local minima** | Can get trapped | Better at escaping |
| **Parallelizable** | Yes (multiple runs) | Limited |
| **Memory** | Low | Higher |
| **Best for** | Large datasets, initial exploration | Final solution, small datasets |

---

## Configuration Parameters

### Common Parameters

| Parameter | Description | Default | Range |
|:----------|:------------|:--------|:------|
| `n_maps` | Number of clusters | `4` | 2-10+ |
| `max_iterations` | Iteration limit | `500` | 100-1000 |
| `clustering_tolerance` | Convergence threshold | `1e-6` | 1e-8 to 1e-4 |
| `n_repeats` | Number of runs | `5` | 1-100 |

### Method Specific

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `initializer` | Initialization method | `Random` | `Random`, `K-Means++` |
| `clustering_method` | Algorithm variant | `Modified K-Means Clustering` | See below |
| `similarity_metric` | Similarity used by the Spatial Similarity variant and by TAAHC | `Spatial Correlation` | `Spatial Correlation`, `Cosine Similarity` |

### Initialization Methods

| Method | Description |
|:-------|:------------|
| `Random` | Random selection of initial templates, drawn without replacement |
| `K-Means++` | Distance-weighted seeding for better spread |

K-Means++ reduces sensitivity to initialization by selecting initial centroids that are well-separated. After a first centre is drawn uniformly at random, each successive seed is sampled with probability proportional to the **squared** polarity-invariant distance to its nearest already-chosen centre, following Arthur & Vassilvitskii (2007):

$$D(x)^2 = \left(1 - \max_{c \in C} |r(x, c)|\right)^2$$

Using the absolute correlation makes the distance polarity-invariant, so a sign-flipped copy of an existing centre is treated as a duplicate rather than as a distant candidate.

{: .note }
> Both initializers draw from the global NumPy random number generator, so setting a random seed makes initialization reproducible for either method.

### Clustering Methods

| Method | Description |
|:-------|:------------|
| `Modified K-Means Clustering` | Polarity-invariant K-means on the raw activation |
| `Modified K-Means Clustering with Spatial Similarity` | Polarity-invariant K-means on `similarity_metric` |
| `Topographic Atomize and Agglomerate Hierarchical Clustering` | Top-down hierarchical (TAAHC) |

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

The best run is the one with the highest **Global Explained Variance (GEV)**, computed on the entire dataset rather than on the clustered subset.

{: .note }
> TAAHC is deterministic, so it always runs exactly once; any larger `n_repeats` setting is ignored.

---

## Configuration Examples

### Standard Analysis

```ini
[clustering_config]
n_maps = 4
initializer = K-Means++
clustering_method = Modified K-Means Clustering
max_iterations = 500
clustering_tolerance = 1e-6
n_repeats = 20
```

### Thorough Exploration

```ini
[clustering_config]
n_maps = 5
initializer = Random
clustering_method = Modified K-Means Clustering with Spatial Similarity
similarity_metric = Spatial Correlation
max_iterations = 1000
clustering_tolerance = 1e-8
n_repeats = 100
```

### TAAHC for Final Solution

```ini
[clustering_config]
n_maps = 4
clustering_method = Topographic Atomize and Agglomerate Hierarchical Clustering
similarity_metric = Spatial Correlation
max_iterations = 500
clustering_tolerance = 1e-6
n_repeats = 1  # TAAHC is deterministic
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

