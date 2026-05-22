---
title: Home
layout: home
nav_order: 1
description: "EEG-COMET: A comprehensive open-source toolbox for EEG microstate analysis"
permalink: /
---

<div class="hero-section">
  <img src="{{ '/assets/images/logo.png' | relative_url }}" alt="EEG-COMET Logo" style="width: 120px; margin-bottom: 1rem;">
  <h1>☄️ EEG-COMET</h1>
  <p><strong>EEG Comprehensive Microstate Extraction Toolbox</strong></p>
  <p>A powerful, open-source platform for comprehensive EEG microstate analysis with automated workflows, machine learning-based classification, and advanced statistical methods.</p>
</div>

---

## What is EEG-COMET?

EEG-COMET is a comprehensive, free and open-source toolbox (released under **GPL-3.0**) that addresses fundamental methodological challenges in EEG microstate analysis. Unlike existing tools that provide only basic functionality, EEG-COMET integrates all essential analytical steps into a unified framework:

- **Quality control and preprocessing**
- **Objective microstate labeling using machine learning**
- **Advanced feature extraction with complexity measures**
- **Source localization for cortical interpretation**

Developed by the [SFU eBrain Lab](https://www.ebrainlab.ca), EEG-COMET eliminates subjective manual processes through machine learning algorithms and provides three distinct analytical approaches to capture brain network dynamics previously obscured by methodological limitations.

---

## Key Features

<div class="feature-grid">
  <div class="feature-card">
    <h3><span class="icon">🔬</span> Comprehensive Framework</h3>
    <p>Integrates quality control, preprocessing, automated assignment, transient segment management, and feature extraction in one unified platform.</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🤖</span> ML-Based Labeling</h3>
    <p>Objective assignment to canonical microstate classes (A-G) using a convolutional neural network with 98%+ accuracy, eliminating subjective visual inspection.</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">📊</span> Three Analysis Modes</h3>
    <p>Traditional full-recording analysis, window-based dynamic tracking, and single-trial event-related analysis for diverse experimental paradigms.</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">📈</span> Advanced Statistics</h3>
    <p>Comprehensive statistical framework including GEE, LMM/GLMM, and cluster-based permutation testing for rigorous inference.</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🧠</span> Source Localization</h3>
    <p>Estimate cortical sources of microstate topographies using multiple inverse-solution methods (dSPM, MNE, sLORETA, eLORETA).</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🖥️</span> Intuitive GUI</h3>
    <p>User-friendly graphical interface accessible to researchers regardless of programming expertise, with cross-platform support.</p>
  </div>
</div>

---

## Modular Architecture

EEG-COMET features ten interconnected processing modules that facilitate comprehensive microstate analysis:

<div class="workflow-container">
  <span class="workflow-step">Data Loading</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Preprocessing</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Data Selection</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Validation</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Clustering</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Labeling</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Backfitting</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Features</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Statistics</span>
  <span class="workflow-arrow">→</span>
  <span class="workflow-step">Sources</span>
</div>

| Module | Description |
|:-------|:------------|
| [Data Loading]({% link modules/data-loading.md %}) | Automated data identification across multiple formats (EEGLAB, Brainstorm, FieldTrip, BIDS) |
| [Data Preparation]({% link modules/data-preparation.md %}) | Zero-phase temporal and spatial filtering optimized for microstate analysis |
| [Data Selection]({% link modules/data-selection.md %}) | GFP peak selection, random subsampling, or all timepoints for clustering |
| [Cluster Validation]({% link modules/cluster-validation.md %}) | 10 statistical methods with 3 selection strategies for optimal cluster numbers |
| [Clustering]({% link modules/clustering.md %}) | Modified K-means and TAAHC algorithms for template identification |
| [Labeling]({% link modules/labeling.md %}) | Automated ML classification or manual labeling of microstate templates |
| [Backfitting]({% link modules/backfitting.md %}) | Template assignment with segment duration optimization and refinement |
| [Feature Extraction]({% link modules/feature-extraction.md %}) | Classical metrics, complexity measures, and event-related dynamics |
| [Statistical Analysis]({% link modules/statistical-analysis.md %}) | Parametric tests, regression frameworks, and permutation-based methods |
| [Source Localization]({% link modules/source-localization.md %}) | Cortical source estimation with standardized or individualized anatomy |

---

## Quick Start

### 1. Install EEG-COMET

```bash
# Clone the repository
git clone https://github.com/eeg-comet/eeg-comet.git
cd eeg-comet

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate eegcomet
```

See the [Installation Guide]({% link installation.md %}) for `pip` and developer installs.

### 2. Launch the GUI

```bash
cd EEG_COMET
python main.py
```

### 3. Load Your Data

Use the GUI to navigate to your preprocessed EEG data directory. EEG-COMET supports multiple formats including `.set`, `.fif`, `.edf`, and BIDS-formatted datasets.

{: .warning }
> **Important:** Ensure proper artifact rejection (especially ocular artifacts) before loading data into EEG-COMET, as artifact contamination can severely compromise microstate analyses.

---

## Documentation Sections

<div class="module-card">
  <h4><a href="{% link installation.md %}">📦 Installation Guide</a></h4>
  <p>Step-by-step installation instructions for Windows, macOS, and Linux using Conda or pip.</p>
</div>

<div class="module-card">
  <h4><a href="{% link getting-started.md %}">🚀 Getting Started</a></h4>
  <p>Quick tutorial to run your first microstate analysis with EEG-COMET.</p>
</div>

<div class="module-card">
  <h4><a href="{% link modules/index.md %}">📚 Module Documentation</a></h4>
  <p>Detailed documentation for each of the 10 processing modules with examples and best practices.</p>
</div>

<div class="module-card">
  <h4><a href="{% link parameters.md %}">⚙️ Parameters Reference</a></h4>
  <p>Complete reference of all configuration parameters with defaults and recommended values.</p>
</div>

---

## Citation

If you use EEG-COMET in a publication, please cite the toolbox. The canonical, machine-readable citation lives in [`CITATION.cff`](https://github.com/eeg-comet/eeg-comet/blob/main/CITATION.cff) at the repository root — GitHub renders a "Cite this repository" button from it. A fallback BibTeX entry:

```bibtex
@software{eegcomet,
  title   = {EEG-COMET: EEG Comprehensive Microstate Extraction Toolbox},
  author  = {Kabir, Amin and {SFU eBrain Lab}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/eeg-comet/eeg-comet},
  license = {GPL-3.0-or-later}
}
```

When a peer-reviewed paper describing the toolbox is available, replace the entry above with the published reference.

---

## License

EEG-COMET is released under the **GNU General Public License v3.0**. You are free to use, study, modify, and redistribute the toolbox; any redistributed or modified version (including a fork or a tool that incorporates EEG-COMET) must also be released under GPL-3.0 with the original copyright and license notices intact. See the full text in [`LICENSE`](https://github.com/eeg-comet/eeg-comet/blob/main/LICENSE).

For closed-source or otherwise GPL-incompatible use, please contact the authors to discuss a separate commercial license.

---

## Support

For questions, bug reports, or contributions:

- **GitHub Issues:** [Report a bug or request a feature](https://github.com/eeg-comet/eeg-comet/issues)
- **Lab Website:** [www.ebrainlab.ca](https://www.ebrainlab.ca)
- **Email:** Contact the development team through the lab website

---

<p style="text-align: center; color: #666; font-size: 0.9rem;">
  EEG-COMET is developed and maintained by the <a href="https://www.ebrainlab.ca">SFU eBrain Lab</a><br>
  Released under the GNU General Public License v3.0
</p>

