---
title: 10. Source Localization
layout: default
parent: Modules
nav_order: 10
description: "Source Localization Module - Cortical source estimation for microstates"
---

# Source Localization Module
{: .no_toc }

Estimate cortical sources of microstate topographies using inverse-solution methods.
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

While microstate topographies describe the spatial distribution of electrical potentials across the scalp, they do not directly identify the underlying cortical sources. Source localization overcomes this limitation by estimating which brain regions contribute to each microstate class.

### Benefits of Source Localization

| Benefit | Description |
|:--------|:------------|
| **Anatomical interpretation** | Link microstates to brain regions |
| **Cross-study comparison** | Independent of electrode setup |
| **Clinical relevance** | Map dysfunction to anatomy |
| **Network understanding** | Identify contributing regions |

---

## Anatomical Frameworks

### Standardized Template MRI

Use a standard brain template for group analyses.

| Advantage | Consideration |
|:----------|:--------------|
| Consistent reference frame | Individual anatomy differences ignored |
| Easy cross-subject comparison | Less precise localization |
| No individual MRI required | Population-average anatomy |

**Best for:** Large group studies, normative comparisons

### Individualized MRI

Use subject-specific MRI for precise localization.

| Advantage | Consideration |
|:----------|:--------------|
| Subject-specific anatomy | Requires individual MRI |
| Accurate localization | More processing steps |
| Clinical precision | FreeSurfer reconstruction needed |

**Best for:** Clinical applications, individual differences research

### FreeSurfer Integration

For individualized analysis, EEG-COMET integrates with FreeSurfer:

1. **Input:** T1-weighted MRI
2. **Process:** `recon-all` pipeline
3. **Output:** Cortical surface, source space

```bash
# FreeSurfer preprocessing (external)
recon-all -s subject_id -i T1.nii -all
```

---

## Inverse Solution Methods

### Minimum Norm Estimate (MNE)

**Basic distributed source method** minimizing overall source power (Hämäläinen & Ilmoniemi, 1994).

$$\hat{J} = R \cdot G^T (G \cdot R \cdot G^T + \lambda C)^{-1} \cdot M$$

| Property | Value |
|:---------|:------|
| Regularization | Depth-weighted |
| Output | Current density |
| Interpretation | Minimum power solution |

### Dynamic Statistical Parametric Mapping (dSPM)

**Noise-normalized MNE** providing statistical maps (Dale et al., 2000).

$$\text{dSPM} = \frac{\text{MNE}}{\sqrt{\text{variance}}}$$

| Property | Value |
|:---------|:------|
| Normalization | Noise variance |
| Output | Z-score-like values |
| Interpretation | Statistical significance |

**Best for:** Most applications, interpretable units

### Standardized Low-Resolution Electromagnetic Tomography (sLORETA)

**Current density normalized** by estimated variance (Pascual-Marqui, 2002).

| Property | Value |
|:---------|:------|
| Normalization | Source variance |
| Output | Standardized values |
| Zero localization error | For single dipoles |

### Exact Low-Resolution Electromagnetic Tomography (eLORETA)

**Improved sLORETA** with exact zero localization error (Pascual-Marqui et al., 2011).

| Property | Value |
|:---------|:------|
| Mathematical property | Exact zero error |
| Smoothness | Higher than sLORETA |
| Computation | More intensive |

### Method Comparison

| Method | Speed | Localization | Smoothness | Statistical |
|:-------|:------|:-------------|:-----------|:------------|
| MNE | Fast | Moderate | Low | No |
| dSPM | Fast | Moderate | Low | Yes |
| sLORETA | Medium | Good | Medium | Partial |
| eLORETA | Slow | Best | High | No |

---

## Source Reconstruction Approaches

### Temporal Averaging Method

Source reconstruction at each timepoint, then aggregated by microstate.

**Algorithm:**

```
1. For each timepoint t:
   - Apply inverse solution
   - Estimate source activation
   
2. For each microstate k:
   - Average source activations across all t assigned to k
   - Result: Mean source map for microstate k
```

| Advantage | Consideration |
|:----------|:--------------|
| Preserves temporal specificity | Computationally intensive |
| Aggregates statistical power | Requires good segmentation |
| Standard approach | Assumes stationarity within state |

### TESS Method

**Topographic Electrophysiological State Source Imaging** - specialized for microstate analysis (Custo et al., 2014, 2017).

**Two-stage approach:**

**Stage 1: Spatial GLM**
- EEG channels as dependent variables
- Estimate source time courses

**Stage 2: Temporal GLM**
- Source time courses as dependent variables
- Microstate templates as predictors
- Identify microstate-correlated sources

| Advantage | Consideration |
|:----------|:--------------|
| Designed for microstates | More complex |
| Statistical framework | Requires permutation testing |
| Accounts for autocorrelation | Computationally intensive |

---

## Source Space Configuration

### Spacing Options

| Setting | Description | Source Count |
|:--------|:------------|:-------------|
| `ico3` | Coarse (fast) | ~1,280 sources |
| `ico4` | Medium | ~5,120 sources |
| `ico5` | Fine (detailed) | ~20,480 sources |

**Recommendation:** `ico3` for quick analyses, `ico4` for publication

### Forward Model

EEG-COMET computes the forward model (leadfield matrix) using:

- **BEM** (Boundary Element Method) for realistic head model
- **Sphere** model as fallback

---

## Configuration

### Parameters

| Parameter | Description | Default | Options |
|:----------|:------------|:--------|:--------|
| `inverse_method` | Source estimation algorithm | `dSPM` | `MNE`, `dSPM`, `sLORETA`, `eLORETA` |
| `source_localization_method` | Reconstruction approach | `tess` | `avg`, `tess` |
| `spacing` | Source space resolution | `ico3` | `ico3`, `ico4`, `ico5` |
| `n_permutations` | Permutations for TESS | `2000` | 1000-10000 |
| `anatomy_subjects_dir` | FreeSurfer subjects directory | `[]` | Path |

### Example Configurations

#### Quick Analysis (Standard Template)

```ini
[source_config]
inverse_method = dSPM
source_localization_method = avg
spacing = ico3
n_permutations = 1000
anatomy_subjects_dir = []
```

#### Detailed Analysis (Individual MRI)

```ini
[source_config]
inverse_method = eLORETA
source_localization_method = tess
spacing = ico4
n_permutations = 5000
anatomy_subjects_dir = /path/to/freesurfer/subjects
```

---

## Coregistration

### Purpose

Align EEG electrode positions with anatomical MRI.

### Automatic Coregistration

EEG-COMET provides automated coregistration:

1. Identify fiducial points (nasion, left/right preauricular)
2. Fit electrodes to scalp surface
3. Verify alignment visually

### Manual Refinement

Fine-tune coregistration through GUI:
- Adjust rotation
- Adjust translation
- Verify electrode positions

---

## Visualization

### Source Maps

EEG-COMET displays:
- Cortical surface with source activation
- Colormap indicating activity strength
- Multiple views (lateral, medial, dorsal)

### 3D Interactive Viewer

Using PyVista:
- Rotate and zoom
- Threshold activity levels
- Compare microstate sources

### Export Options

| Format | Use |
|:-------|:----|
| PNG/SVG | Publication figures |
| NIfTI | Integration with other tools |
| STC | MNE-Python source time courses |

---

## Interpretation

### Microstate-Network Associations

Canonical microstates have been associated with specific networks (Britz et al., 2010; Custo et al., 2017):

| Microstate | Associated Regions | Networks |
|:-----------|:-------------------|:---------|
| **A** | Superior temporal, inferior frontal | Auditory, language |
| **B** | Occipital, parietal | Visual processing |
| **C** | Anterior cingulate, insula | Salience, default mode |
| **D** | Frontal, parietal | Attention, executive |

{: .note }
> These associations are based on normative data and may vary in clinical populations.

### Cautionary Notes

1. **Inverse problem is ill-posed** - Multiple source configurations can produce same scalp topography
2. **Regularization assumptions** - Affect localization precision
3. **Individual variability** - Template-based results are approximate
4. **Deep sources** - EEG has limited sensitivity to subcortical generators

---

## Workflow

### Standard Analysis Workflow

```
1. Complete microstate extraction and labeling
2. Configure source localization parameters
3. Set up coregistration (if individual MRI)
4. Run source estimation
5. Visualize and export results
```

### Quality Checks

| Check | Action |
|:------|:-------|
| Coregistration accuracy | Verify electrode positions on scalp |
| Forward model | Check for warnings |
| Source coverage | Ensure brain regions covered |
| Activation patterns | Compare to literature |

---

## Best Practices

1. **Match method to precision needs**
   - dSPM for general use
   - eLORETA for maximum precision

2. **Use individual MRI when possible**
   - Critical for clinical applications
   - Better localization accuracy

3. **Verify coregistration**
   - Visual inspection essential
   - Check fiducial placement

4. **Consider depth sensitivity**
   - EEG best for cortical sources
   - Deep sources less reliable

5. **Report limitations**
   - Acknowledge inverse problem
   - Note template vs. individual anatomy

---

## Troubleshooting

<div class="callout warning">
<strong>Forward model computation fails</strong><br>
Check electrode positions are reasonable. Ensure all required files are present. Try sphere model as fallback.
</div>

<div class="callout warning">
<strong>Source maps look noisy</strong><br>
Increase regularization. Check EEG data quality. Consider smoother method (sLORETA).
</div>

<div class="callout warning">
<strong>Coregistration doesn't align</strong><br>
Verify fiducial points are correctly identified. Manually adjust translation/rotation. Check electrode coordinate system.
</div>

<div class="callout warning">
<strong>FreeSurfer directory not found</strong><br>
Set SUBJECTS_DIR environment variable. Verify freesurfer preprocessing completed. Check path in configuration.
</div>

---

## References

Key methodological references for source localization:

- Hämäläinen, M. S., & Ilmoniemi, R. J. (1994). Interpreting magnetic fields of the brain: minimum norm estimates. *Medical & Biological Engineering & Computing*, 32(1), 35–42. [https://doi.org/10.1007/BF02512476](https://doi.org/10.1007/BF02512476)
- Dale, A. M., Liu, A. K., Fischl, B. R., Buckner, R. L., Belliveau, J. W., Lewine, J. D., & Halgren, E. (2000). Dynamic statistical parametric mapping: Combining fMRI and MEG for high-resolution imaging of cortical activity. *Neuron*, 26(1), 55–67. [https://doi.org/10.1016/S0896-6273(00)81138-1](https://doi.org/10.1016/S0896-6273(00)81138-1)
- Pascual-Marqui, R. D. (2002). Standardized low-resolution brain electromagnetic tomography (sLORETA): technical details. *Methods and Findings in Experimental and Clinical Pharmacology*, 24(Suppl D), 5–12.
- Pascual-Marqui, R. D., Lehmann, D., Koukkou, M., Kochi, K., Anderer, P., Saletu, B., Tanaka, H., Hirata, K., John, E. R., Prichep, L., Biscay-Lirio, R., & Kinoshita, T. (2011). Assessing interactions in the brain with exact low-resolution electromagnetic tomography (eLORETA). *Philosophical Transactions of the Royal Society A*, 369(1952), 3768–3784. [https://doi.org/10.1098/rsta.2011.0081](https://doi.org/10.1098/rsta.2011.0081)
- Michel, C. M., Murray, M. M., Lantz, G., Gonzalez, S., Spinelli, L., & Grave de Peralta, R. (2004). EEG source imaging. *Clinical Neurophysiology*, 115(10), 2195–2222. [https://doi.org/10.1016/j.clinph.2004.06.001](https://doi.org/10.1016/j.clinph.2004.06.001)
- Custo, A., Vulliemoz, S., Grouiller, F., Van De Ville, D., & Michel, C. M. (2014). EEG source imaging of brain states using spatiotemporal regression. *NeuroImage*, 96, 106–116. [https://doi.org/10.1016/j.neuroimage.2014.04.002](https://doi.org/10.1016/j.neuroimage.2014.04.002)
- Custo, A., Van De Ville, D., Wells, W. M., Tomescu, M. I., Brunet, D., & Michel, C. M. (2017). Electroencephalographic resting-state networks: Source localization of microstates. *Brain Connectivity*, 7(10), 671–682. [https://doi.org/10.1089/brain.2016.0476](https://doi.org/10.1089/brain.2016.0476)
- Britz, J., Van De Ville, D., & Michel, C. M. (2010). BOLD correlates of EEG topography reveal rapid resting-state network dynamics. *NeuroImage*, 52(4), 1162–1170. [https://doi.org/10.1016/j.neuroimage.2010.02.052](https://doi.org/10.1016/j.neuroimage.2010.02.052)
- Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. [https://doi.org/10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)

---

## Summary

The Source Localization module completes the EEG-COMET pipeline by linking scalp topographies to cortical sources, enabling neuroanatomical interpretation of microstate dynamics.

[**← Back to Modules Overview**]({% link modules/index.md %})

