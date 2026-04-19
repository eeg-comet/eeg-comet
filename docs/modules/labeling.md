---
title: 6. Labeling
layout: default
parent: Modules
nav_order: 6
description: "Microstate Labeling Module - Automated and manual template classification"
---

# Microstate Labeling Module
{: .no_toc }

Automated ML-based classification and manual labeling of microstate templates.
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

Following microstate template extraction, each identified topography must be labeled to enable cross-study comparisons and facilitate interpretation. EEG-COMET provides two complementary approaches:

1. **Automated ML-Based Classification** - CNN-based labeling achieving 98%+ accuracy
2. **Manual Labeling** - Expert visual inspection and assignment

{: .important }
> Inconsistent labeling conventions have severely limited cross-study comparisons. EEG-COMET's automated classifier provides objective, reproducible labeling aligned with consensus nomenclature.

---

## Canonical Microstate Classes

The established canonical microstates identified in normative studies:

| Class | Topography | Associated Networks/Functions |
|:------|:-----------|:-----------------------------|
| **A** | Left-right orientation | Auditory processing, phonological networks |
| **B** | Right-left orientation | Visual processing networks |
| **C** | Anterior-posterior orientation | Salience, interoception, default mode |
| **D** | Frontocentral pattern | Attention, executive control |
| **E** | Central pattern | Motor and sensorimotor networks |
| **F** | Frontal focus | Executive and cognitive control |
| **G** | Posterior focus | Visual and attentional networks |

{: .note }
> Classes A-D are the most commonly reported. Classes E-G emerge when analyzing more than 4 microstates.

---

## Automated Classification

### CNN Architecture

The classifier uses a **Convolutional Neural Network** with position-dependent learning to capture hierarchical spatial dependencies in topographic patterns.

**Key Design Principles:**

| Feature | Rationale |
|:--------|:----------|
| Position-dependent | Maintains spatial orientation relative to anatomy |
| Not rotation-invariant | Different orientations = different microstates |
| Montage-independent | Standardized topographic image input |

### Preprocessing for Classification

1. **Interpolation**: Convert template to standardized 2D grid
2. **Normalization**: Scale values to consistent range
3. **Alignment**: Map to common coordinate system

### Training Data

| Dataset Characteristic | Value |
|:-----------------------|:------|
| Subjects | 1,157 |
| Sources | Multiple independent studies |
| Labels | Expert-validated |
| Accuracy | >98% on validation |

### Supported Classes

The classifier identifies up to **7 canonical classes**: A, B, C, D, E, F, G

For analyses with fewer clusters, only relevant labels are assigned.

---

## Using Automated Classification

### GUI Workflow

1. Complete microstate clustering
2. Click **"Auto-Label"** button
3. Review assigned labels in visualization panel
4. Verify assignments match expected topographies

### Confidence Scores

The classifier provides confidence scores for each assignment:

| Confidence | Interpretation | Action |
|:-----------|:---------------|:-------|
| >0.95 | High confidence | Accept |
| 0.80-0.95 | Moderate confidence | Verify visually |
| <0.80 | Low confidence | Manual review recommended |

### Handling Atypical Topographies

If a template doesn't match any canonical class well:

- Classifier assigns best match with low confidence
- Warning is displayed
- Consider manual labeling
- May indicate novel or pathological patterns

---

## Manual Labeling

### When to Use

| Scenario | Recommendation |
|:---------|:---------------|
| Atypical topographies | Classifier may not recognize |
| Clinical populations | Patterns may differ from normative |
| Exploratory analyses | Beyond canonical 7 classes |
| Validation | Confirm automated labels |

### Workflow

1. View each template topography
2. Compare with canonical reference maps
3. Assign labels based on visual similarity
4. Document reasoning for assignments

### Reference Topographies

EEG-COMET displays canonical topographies for comparison:

- Standard A, B, C, D patterns
- Extended E, F, G patterns
- Polarity-inverted versions

### Custom Labels

For exploratory analyses, custom labels can be assigned:

- Numeric labels (1, 2, 3, ...)
- Descriptive labels (Frontal, Posterior, ...)
- Study-specific nomenclature

---

## Polarity Considerations

### Polarity Invariance

Microstates differing only in voltage sign represent the same neural configuration:

- **Microstate A** and **Microstate A (inverted)** are equivalent
- Classifier handles both polarities

### Display Convention

EEG-COMET follows the convention of showing:
- Positive values in warm colors (red/orange)
- Negative values in cool colors (blue)
- Consistent polarity within a study

---

## Cross-Study Comparability

### Benefits of Standardized Labeling

| Benefit | Description |
|:--------|:------------|
| Meta-analysis | Combine findings across studies |
| Replication | Direct comparison with published results |
| Clinical norms | Compare patients to healthy reference |
| Longitudinal | Track changes over time |

### Ensuring Comparability

1. **Use automated classifier** for consistent labeling
2. **Report classifier version** in publications
3. **Include topographic figures** for visual verification
4. **Document any manual overrides** and rationale

---

## Configuration

### Automated Classification Settings

Classification is typically automatic with default settings. Advanced options:

| Setting | Description |
|:--------|:------------|
| Model file | `model_v2.onnx` (bundled in `EEG_COMET/models/`) |
| Inference runtime | ONNX Runtime (`onnxruntime`) |
| Confidence threshold | Report low-confidence assignments |
| Interpolation grid | Standard 64×64 pixels |

### Manual Override

After automated classification:

1. Click on any template
2. Select new label from dropdown
3. Changes are logged for reproducibility

---

## Quality Assurance

### Verification Steps

1. **Visual check**: Do assigned labels match topographic patterns?
2. **Consistency**: Are similar topographies labeled consistently across subjects?
3. **Confidence review**: Investigate low-confidence assignments

### Common Issues

<div class="callout warning">
<strong>Same label assigned to multiple templates</strong><br>
This may occur if topographies are very similar. Consider reducing K or merging clusters.
</div>

<div class="callout warning">
<strong>Unexpected label assignment</strong><br>
Verify topography is artifact-free. Check if non-canonical pattern requires manual labeling.
</div>

<div class="callout warning">
<strong>Low confidence on all assignments</strong><br>
May indicate unusual electrode montage or data quality issues. Verify preprocessing.
</div>

---

## Best Practices

1. **Always verify automated labels**
   - Even with high confidence
   - Compare to canonical patterns

2. **Document manual changes**
   - Record rationale
   - Note in methods section

3. **Be consistent within studies**
   - Same labeling approach for all subjects
   - Don't mix automated and manual

4. **Report classifier details**
   - Version information
   - Confidence thresholds used

5. **Consider domain knowledge**
   - Clinical expertise may reveal meaningful patterns
   - Novel paradigms may produce atypical states

---

## Output

### Label Information

For each microstate template:

| Field | Description |
|:------|:------------|
| `label` | Assigned class (A, B, C, etc.) |
| `confidence` | Classification confidence (0-1) |
| `method` | 'automated' or 'manual' |
| `original_index` | Cluster index before labeling |

### Reordering

Templates are typically reordered after labeling:
- Alphabetical by label (A, B, C, D, ...)
- Maintains consistent presentation

---

## Next Step

[**Template Backfitting Module →**]({% link modules/backfitting.md %})

