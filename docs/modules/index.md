---
title: Modules
layout: default
nav_order: 4
has_children: true
description: "Detailed documentation for EEG-COMET processing modules"
---

# Processing Modules
{: .no_toc }

EEG-COMET features ten interconnected processing modules that provide comprehensive microstate analysis capabilities.
{: .fs-6 .fw-300 }

---

## Module Overview

The modular architecture allows researchers to customize their analysis pipeline while maintaining methodological rigor. Each module addresses specific challenges in microstate analysis.

```mermaid
flowchart LR
    A([1. Data&nbsp;Loading]) --> B([2. Preprocessing])
    B --> C([3. Data&nbsp;Selection])
    C --> D([4. Validation])
    D --> E([5. Clustering])
    E --> F([6. Labeling])
    F --> G([7. Backfitting])
    G --> H([8. Features])
    H --> I([9. Statistics])
    I --> J([10. Sources])

    classDef comet fill:#8B1538,stroke:#6B1028,color:#ffffff,stroke-width:1px,rx:6,ry:6;
    class A,B,C,D,E,F,G,H,I,J comet;
```

---

## Module Descriptions

### Data Acquisition & Preparation

<div class="module-card">
  <h4><a href="{% link modules/data-loading.md %}">📂 1. Data Loading Module</a></h4>
  <p>Automated data identification across multiple formats (EEGLAB, Brainstorm, FieldTrip, BIDS) with recursive directory searching and electrode configuration verification.</p>
</div>

<div class="module-card">
  <h4><a href="{% link modules/data-preparation.md %}">🔧 2. Data Preparation Module</a></h4>
  <p>Zero-phase temporal filtering (FIR/IIR) and optional spatial filtering using k-nearest-neighbor interpolation to optimize data for microstate analysis.</p>
</div>

<div class="module-card">
  <h4><a href="{% link modules/data-selection.md %}">📍 3. Clustering Data Selection Module</a></h4>
  <p>Three complementary methods for selecting which timepoints contribute to template extraction: GFP peaks, random subsampling, or all timepoints.</p>
</div>

### Template Identification

<div class="module-card">
  <h4><a href="{% link modules/cluster-validation.md %}">📊 4. Cluster Number Validation Module</a></h4>
  <p>Comprehensive validation framework with 10 statistical methods and 3 selection strategies to determine optimal microstate cluster numbers.</p>
</div>

<div class="module-card">
  <h4><a href="{% link modules/clustering.md %}">🎯 5. Microstate Clustering Module</a></h4>
  <p>Modified K-means and TAAHC algorithms for identifying microstate templates, with configurable convergence criteria and multiple initialization strategies.</p>
</div>

<div class="module-card">
  <h4><a href="{% link modules/labeling.md %}">🏷️ 6. Microstate Labeling Module</a></h4>
  <p>Automated ML-based classification achieving 98%+ accuracy for canonical microstate labels (A-G), plus manual labeling options for exploratory analyses.</p>
</div>

### Segmentation & Analysis

<div class="module-card">
  <h4><a href="{% link modules/backfitting.md %}">✂️ 7. Template Backfitting Module</a></h4>
  <p>Template assignment via spatial correlation with automated or manual segment duration optimization and four strategies for handling short segments.</p>
</div>

<div class="module-card">
  <h4><a href="{% link modules/feature-extraction.md %}">📈 8. Feature Extraction Module</a></h4>
  <p>Classical temporal metrics (COV, OCC, DUR, GEV), transition probabilities, complexity measures (entropy, Lempel-Ziv, Hurst), and event-related dynamics.</p>
</div>

### Statistical Inference & Interpretation

<div class="module-card">
  <h4><a href="{% link modules/statistical-analysis.md %}">🧮 9. Statistical Analysis Module</a></h4>
  <p>Parametric tests for averaged metrics, GEE/LMM regression for trial-level data, and cluster-based permutation testing for temporal dynamics.</p>
</div>

<div class="module-card">
  <h4><a href="{% link modules/source-localization.md %}">🧠 10. Source Localization Module</a></h4>
  <p>Cortical source estimation using multiple inverse methods (dSPM, MNE, sLORETA, eLORETA) with standardized or individualized anatomical models.</p>
</div>

---

## Choosing Your Workflow

### Resting-State Analysis

For traditional resting-state microstate analysis:

1. **Data Loading** → Load artifact-free continuous recordings
2. **Data Preparation** → Bandpass filter 2-20 Hz (FIR)
3. **Data Selection** → GFP peaks for traditional approach
4. **Cluster Validation** → Majority vote across 10 criteria
5. **Clustering** → Modified K-means with multiple repeats
6. **Labeling** → Automated CNN classification
7. **Backfitting** → Segment exclusion for short segments
8. **Feature Extraction** → Static mode, all standard metrics

### Event-Related Analysis

For studying microstate dynamics around experimental events:

1. **Data Loading** → Load epoched data with event markers
2. **Data Preparation** → Bandpass filter 1-40 Hz (FIR)
3. **Data Selection** → All timepoints for complete coverage
4. **Cluster Validation** → Focus on GEV and CV criteria
5. **Clustering** → TAAHC for thorough exploration
6. **Labeling** → Automated or manual as appropriate
7. **Backfitting** → Smooth strategy to preserve temporal structure
8. **Feature Extraction** → Event-related mode with ROF/RTF metrics
9. **Statistical Analysis** → Cluster-based permutation testing

### Clinical/Comparative Studies

For comparing groups or conditions:

1. Follow resting-state workflow through Feature Extraction
2. **Statistical Analysis** → t-tests with FDR correction
3. **Source Localization** → Map significant differences to cortex

---

## Best Practices

{: .highlight }
> **Preprocessing First:** Always complete artifact rejection before using EEG-COMET. Ocular and muscle artifacts create spurious topographies that contaminate all downstream analyses.

{: .note }
> **Consistent Parameters:** Use identical parameters across all subjects/conditions within a study for valid comparisons.

{: .important }
> **Document Everything:** EEG-COMET automatically logs all processing steps. Save these logs with your results for reproducibility.

