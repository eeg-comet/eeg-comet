---
title: Home
layout: home
nav_order: 1
description: "EEG-COMET: A comprehensive open-source toolbox for resting-state and event-related EEG microstate analysis."
permalink: /
---

<div class="hero-section" markdown="0">
  <img src="{{ '/assets/images/logo.png' | relative_url }}" alt="EEG-COMET Logo" style="width: 120px; margin-bottom: 1rem;">
  <h1>☄️ EEG-COMET</h1>
  <p><strong>EEG Comprehensive Microstate Extraction Toolbox</strong></p>
  <p>An open-source platform for end-to-end EEG microstate analysis &mdash; preprocessing, automated cluster validation, ML-based labeling, backfitting, feature extraction, statistics, and source localization, in a single reproducible workflow.</p>
</div>

[Get started]({% link getting-started.md %}){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Install]({% link installation.md %}){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/eeg-comet/eeg-comet){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What is EEG-COMET?

EEG-COMET is a comprehensive, free, and open-source toolbox (released
under **GPL-3.0**) that addresses fundamental methodological challenges
in EEG microstate analysis. Where most existing tools provide only
fragments of the pipeline, EEG-COMET integrates every analytical step
into a single, reproducible framework:

- **Quality control & preprocessing** &mdash; consistent, zero-phase
  filtering and channel handling tuned for microstate analysis.
- **Automated cluster validation** &mdash; ten statistical criteria and
  three selection strategies for choosing the number of microstates.
- **ML-based labeling** &mdash; an embedded CNN classifier assigns
  canonical microstate labels (A&ndash;G) with **>98% accuracy**,
  removing subjective visual inspection.
- **Advanced feature extraction** &mdash; classical metrics, transition
  probabilities, complexity measures, and event-related dynamics.
- **Source localization** &mdash; cortical estimation via dSPM, MNE,
  sLORETA, and eLORETA on standardized or individualized anatomy.

Developed and maintained by the [SFU eBrain Lab](https://www.ebrainlab.ca)
at Simon Fraser University.

{: .new }
> EEG-COMET supports three complementary analysis modes from a single
> pipeline: **full-recording** resting-state analysis, **sliding-window**
> dynamic tracking, and **single-trial event-related** analysis.

---

## Key features

<div class="feature-grid" markdown="0">
  <div class="feature-card">
    <h3><span class="icon">🔬</span> Comprehensive framework</h3>
    <p>Quality control, preprocessing, clustering, labeling, backfitting, features, statistics, and source localization &mdash; one tool, one workflow.</p>
  </div>

  <div class="feature-card">
    <h3><span class="icon">🤖</span> ML-based labeling</h3>
    <p>Objective assignment to canonical classes (A&ndash;G) via a position-aware CNN trained on 1,157 subjects. &gt;98% validation accuracy.</p>
  </div>

  <div class="feature-card">
    <h3><span class="icon">📊</span> Three analysis modes</h3>
    <p>Full-recording, sliding-window dynamic, and single-trial event-related microstate analysis from a single configuration.</p>
  </div>

  <div class="feature-card">
    <h3><span class="icon">📈</span> Advanced statistics</h3>
    <p>Parametric tests, GEE and LMM/GLMM for trial-level data, and cluster-based permutation testing for temporal dynamics.</p>
  </div>

  <div class="feature-card">
    <h3><span class="icon">🧠</span> Source localization</h3>
    <p>Estimate cortical sources of microstate topographies with dSPM, MNE, sLORETA, or eLORETA on standardized or individual anatomy.</p>
  </div>

  <div class="feature-card">
    <h3><span class="icon">🖥️</span> GUI &amp; CLI</h3>
    <p>Point-and-click workflows for new users; a scriptable CLI for batch processing and HPC use. Cross-platform on Windows, macOS, and Linux.</p>
  </div>
</div>

---

## The pipeline

EEG-COMET is organized as nine sequential pipeline modules plus a
study-comparison module for statistics. You can run the full pipeline
end-to-end or use individual modules independently.

<div class="workflow-container" markdown="0">
  <span class="workflow-step">Data Loading</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Preparation</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Data Selection</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Validation</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Clustering</span>
</div>
<div class="workflow-container" markdown="0">
  <span class="workflow-step">Labeling</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Backfitting</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Features</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Sources</span>
</div>

| # | Module | Summary |
|:--|:-------|:--------|
| 1 | [Data Loading]({% link modules/data-loading.md %}) | Automated import across EEGLAB, EDF/BDF/GDF, BrainVision, Neuroscan, EGI, and BIDS. |
| 2 | [Data Preparation]({% link modules/data-preparation.md %}) | Zero-phase temporal filtering and optional k-NN spatial smoothing. |
| 3 | [Data Selection]({% link modules/data-selection.md %}) | GFP peak selection, random subsampling, or all timepoints. |
| 4 | [Cluster Validation]({% link modules/cluster-validation.md %}) | Ten statistical criteria with three selection strategies. |
| 5 | [Clustering]({% link modules/clustering.md %}) | Modified K-means (with an optional spatial-similarity variant) and TAAHC algorithms for template identification. |
| 6 | [Labeling]({% link modules/labeling.md %}) | CNN-based automated labeling (A&ndash;G) or manual assignment. |
| 7 | [Backfitting]({% link modules/backfitting.md %}) | Template assignment with configurable segment-duration handling. |
| 8 | [Feature Extraction]({% link modules/feature-extraction.md %}) | Classical metrics, transitions, complexity measures, ERP dynamics. |
| 9 | [Statistical Analysis]({% link modules/statistical-analysis.md %}) | Run outside the pipeline, from the **Study Comparison and Statistical Analysis** window: parametric tests, GEE/LMM regression, cluster permutation. |
| 10 | [Source Localization]({% link modules/source-localization.md %}) | Cortical source estimation with multiple inverse methods. |

---

## Quick start

### 1. Install

```bash
git clone https://github.com/eeg-comet/eeg-comet.git
cd eeg-comet
conda env create -f environment.yml
conda activate eeg-comet
```

See the [Installation Guide]({% link installation.md %}) for `pip` and
developer (editable) installs.

### 2. Launch the GUI

After installing the package (the conda recipe above already does this, or run
`pip install -e .`), launch the GUI from any directory:

```bash
eeg-comet
```

Equivalently, run the package module directly with `python -m eeg_comet.main`.
A `eeg-comet-cli` terminal entry point is also available.

### 3. Load preprocessed data

Use the GUI to point at your preprocessed EEG data directory. EEG-COMET
supports `.set`, `.edf`, `.bdf`, `.gdf`, BrainVision, Neuroscan, EGI,
Nicolet, eXimia, Persyst, and BIDS datasets.

{: .warning }
> **Apply artifact rejection first.** EEG-COMET expects clean,
> artifact-rejected input. Residual ocular, muscle, or cardiac
> artifacts will contaminate clustering and propagate through every
> downstream metric.

Full walkthrough: [Getting Started]({% link getting-started.md %}).

---

## Documentation map

<div class="module-card" markdown="0">
  <h4><a href="{% link installation.md %}">📦 Installation Guide</a></h4>
  <p>Step-by-step instructions for Windows, macOS, and Linux using Conda, pip, or an editable developer install.</p>
</div>

<div class="module-card" markdown="0">
  <h4><a href="{% link getting-started.md %}">🚀 Getting Started</a></h4>
  <p>A guided walkthrough of running your first microstate analysis end-to-end.</p>
</div>

<div class="module-card" markdown="0">
  <h4><a href="{% link modules/index.md %}">📚 Module Documentation</a></h4>
  <p>Detailed reference for each of the ten processing modules with parameters, examples, and best practices.</p>
</div>

<div class="module-card" markdown="0">
  <h4><a href="{% link parameters.md %}">⚙️ Parameters Reference</a></h4>
  <p>Every configuration parameter, with defaults, valid ranges, and recommended values.</p>
</div>

<div class="module-card" markdown="0">
  <h4><a href="{% link faq.md %}">❓ FAQ</a></h4>
  <p>Common questions about data requirements, analysis choices, and troubleshooting.</p>
</div>

---

## Citation

If you use EEG-COMET in academic work, please cite the companion paper.
The canonical, machine-readable citation lives in
[`CITATION.cff`](https://github.com/eeg-comet/eeg-comet/blob/stable/CITATION.cff)
at the repository root &mdash; GitHub renders a "Cite this repository"
button from it.

BibTeX:

```bibtex
@article{eegcomet,
  title   = {EEG-COMET: A Comprehensive Platform for Resting-State and Single-Trial Event-Related Microstate Analysis},
  author  = {Kabir, Amin and Tarailis, Povilas and Chatterjee, Raaj and Dhami, Prabhjot and Farzan, Faranak},
  year    = {2026},
  url     = {https://github.com/eeg-comet/eeg-comet}
}
```

Once the paper is published, the entry above will be updated with the
journal reference and DOI.

---

## License

EEG-COMET is released under the **GNU General Public License v3.0**.
You are free to use, study, modify, and redistribute it. Any
redistributed or modified version (including a fork or a tool that
incorporates EEG-COMET) must also be released under GPL-3.0 with the
original copyright and license notices intact. See the full text in
[`LICENSE`](https://github.com/eeg-comet/eeg-comet/blob/stable/LICENSE).

For closed-source or otherwise GPL-incompatible use, please contact the
authors to discuss a separate commercial license.

---

## Support &amp; community

- **Bug reports / feature requests:** [GitHub Issues](https://github.com/eeg-comet/eeg-comet/issues)
- **Lab website:** [www.ebrainlab.ca](https://www.ebrainlab.ca)
- **Contact:** through the lab website

---

<p style="text-align: center; color: #666; font-size: 0.9rem;">
  EEG-COMET is developed and maintained by the <a href="https://www.ebrainlab.ca">SFU eBrain Lab</a>,
  Simon Fraser University.<br>
  Released under the GNU General Public License v3.0.
</p>
