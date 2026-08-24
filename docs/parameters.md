---
title: Parameters Reference
layout: default
nav_order: 5
description: "Complete reference of all EEG-COMET configuration parameters"
---

# Parameters Reference
{: .no_toc }

Complete reference of all configuration parameters with defaults and recommended values.
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

## Configuration File Format

EEG-COMET uses INI-format configuration files with sections for each processing stage.

```ini
[section_name]
parameter_name = value
another_parameter = value
```

### Example Complete Configuration

```ini
[io_config]
study_name = my_microstate_study
input_folder = /path/to/eeg/data
output_folder = /path/to/results
channel_location_dir = /path/to/montage.csv
extension = .set
pattern_content = *
load_all_files = True
data_type = raw

[preprocessing_config]
temporal_filter_data = True
filter_method = fir
lowcut_freq = 2
highcut_freq = 20
downsample_data = True
sampling_rate = 250
spatial_filter_data = False
remove_channels = False
channels_to_remove = []

[clustering_config]
smoothing_gfp = False
smoothing_distance = 10
n_maps = 4
k_min = 2
k_max = 10
stopping_mode = majority_vote
stopping_threshold = 10
data_percentage = 100
initializer = Random
clustering_method = Modified K-Means Clustering
max_iterations = 500
clustering_tolerance = 1e-6
n_repeats = 5

[backfitting_config]
backfit_to = all
identify_short_window = False
filter_segments = False
filter_segments_less_than = 20
filter_segments_option = smooth
convergence_epsilon = 1e-6
half_window_size = 3
smoothness_penalty = 5

[features_config]
export_format = .csv
feature_list = OCC, DUR, COV
feature_mode = averaged
feature_types = real
sliding_window_size = 1
duration_method = geometric

[source_config]
inverse_method = dSPM
source_localization_method = tess
n_permutations = 2000
spacing = ico3
anatomy_subjects_dir = []
```

---

## I/O Configuration

### `[io_config]`

Parameters for data input and output paths.

<table class="param-table">
<thead>
<tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr>
</thead>
<tbody>
<tr>
<td><code>study_name</code></td>
<td>String</td>
<td><em>Required</em></td>
<td>Name for this analysis study. Used as output folder name.</td>
</tr>
<tr>
<td><code>input_folder</code></td>
<td>Path</td>
<td><em>Required</em></td>
<td>Directory containing EEG data files. Searched recursively.</td>
</tr>
<tr>
<td><code>output_folder</code></td>
<td>Path</td>
<td><em>Required</em></td>
<td>Directory for saving all output files.</td>
</tr>
<tr>
<td><code>channel_location_dir</code></td>
<td>Path</td>
<td><code>[]</code></td>
<td>Path to electrode location file. Empty if embedded in data.</td>
</tr>
<tr>
<td><code>extension</code></td>
<td>String</td>
<td><code>.auto</code></td>
<td>File extension to search for. Options: <code>.auto</code>, <code>.set</code>, <code>.fif</code>, <code>.edf</code>, etc.</td>
</tr>
<tr>
<td><code>pattern_content</code></td>
<td>String</td>
<td><code>*</code></td>
<td>Filename pattern filter. Use <code>*</code> for wildcards. Ignored when <code>load_all_files = True</code>.</td>
</tr>
<tr>
<td><code>load_all_files</code></td>
<td>Boolean</td>
<td><code>True</code></td>
<td>Load every matching file in the input folder. When <code>False</code>, only files whose name contains <code>pattern_content</code> are loaded. Also read from <code>[preprocessing_config]</code> for configs written by older versions.</td>
</tr>
<tr>
<td><code>data_type</code></td>
<td>String</td>
<td><code>raw</code></td>
<td>Data type: <code>raw</code> (continuous) or <code>epoched</code> (segmented).</td>
</tr>
</tbody>
</table>

### Extension Values

| Value | Description |
|:------|:------------|
| `.auto` | Automatically detect all supported formats |
| `.set` | EEGLAB format |
| `.fif` | MNE-Python format |
| `.edf` | European Data Format |
| `.vhdr` | BrainVision format |
| `.cnt` | Neuroscan format |
| `.mff` | EGI format |

---

## Preprocessing Configuration

### `[preprocessing_config]`

Parameters for data preparation and filtering.

<table class="param-table">
<thead>
<tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr>
</thead>
<tbody>
<tr>
<td><code>temporal_filter_data</code></td>
<td>Boolean</td>
<td><code>True</code></td>
<td>Apply temporal bandpass filtering.</td>
</tr>
<tr>
<td><code>filter_method</code></td>
<td>String</td>
<td><code>fir</code></td>
<td>Filter type: <code>fir</code> or <code>iir</code>.</td>
</tr>
<tr>
<td><code>lowcut_freq</code></td>
<td>Float</td>
<td><code>2</code></td>
<td>High-pass filter cutoff frequency (Hz).</td>
</tr>
<tr>
<td><code>highcut_freq</code></td>
<td>Float</td>
<td><code>20</code></td>
<td>Low-pass filter cutoff frequency (Hz).</td>
</tr>
<tr>
<td><code>downsample_data</code></td>
<td>Boolean</td>
<td><code>True</code></td>
<td>Apply downsampling to reduce data size.</td>
</tr>
<tr>
<td><code>sampling_rate</code></td>
<td>Integer</td>
<td><code>250</code></td>
<td>Target sampling rate (Hz) after downsampling.</td>
</tr>
<tr>
<td><code>spatial_filter_data</code></td>
<td>Boolean</td>
<td><code>False</code></td>
<td>Apply spatial smoothing by averaging each electrode with its nearest neighbors. The neighborhood size is determined adaptively (3-8 neighbors) and is not configurable.</td>
</tr>
<tr>
<td><code>remove_channels</code></td>
<td>Boolean</td>
<td><code>False</code></td>
<td>Remove specified channels.</td>
</tr>
<tr>
<td><code>channels_to_remove</code></td>
<td>List</td>
<td><code>[]</code></td>
<td>Channel names to remove (comma-separated).</td>
</tr>
</tbody>
</table>

### Filter Method Comparison

| Method | Type | Best For |
|:-------|:-----|:---------|
| `fir` | Finite Impulse Response | Event-related, precise timing |
| `iir` | Infinite Impulse Response | Long recordings, efficiency |

### Recommended Frequency Bands

| Band | Range | Use Case |
|:-----|:------|:---------|
| Standard | 2-20 Hz | Traditional microstate analysis |
| Broad | 1-40 Hz | Event-related, preserve oscillations |
| Narrow | 4-15 Hz | Focus on microstate frequencies |

### Frequency Constraints

Filtering runs before resampling so that the low-pass acts as the anti-aliasing filter. The passband must therefore fit below the Nyquist frequency of both the recording and the target rate:

| Constraint | Rule |
|:-----------|:-----|
| Nyquist | `highcut_freq` must be strictly below half of the smaller of the current sampling rate and `sampling_rate` |
| Ordering | `lowcut_freq` must be below `highcut_freq` |

Violating either rule raises a `ValueError` before any data are filtered. The message states the offending value and how to resolve it, either by lowering `highcut_freq` or by raising `sampling_rate` above twice `highcut_freq`.

---

## Clustering Configuration

### `[clustering_config]`

Parameters for data selection, validation, and template extraction.

<table class="param-table">
<thead>
<tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr>
</thead>
<tbody>
<tr>
<td><code>smoothing_gfp</code></td>
<td>Boolean</td>
<td><code>False</code></td>
<td>Enforce a minimum spacing between detected GFP peaks. When <code>False</code>, every local GFP maximum is a candidate peak.</td>
</tr>
<tr>
<td><code>smoothing_distance</code></td>
<td>Integer</td>
<td><code>10</code></td>
<td>Minimum spacing between GFP peaks, in milliseconds. Converted to samples using the sampling rate. Used only when <code>smoothing_gfp = True</code>.</td>
</tr>
<tr>
<td><code>n_maps</code></td>
<td>Integer/String</td>
<td><code>4</code></td>
<td>Number of microstate classes. Use <code>auto</code> for automatic selection.</td>
</tr>
<tr>
<td><code>k_min</code></td>
<td>Integer</td>
<td><code>2</code></td>
<td>Minimum K for automatic selection.</td>
</tr>
<tr>
<td><code>k_max</code></td>
<td>Integer</td>
<td><code>10</code></td>
<td>Maximum K for automatic selection.</td>
</tr>
<tr>
<td><code>stopping_mode</code></td>
<td>String</td>
<td><code>majority_vote</code></td>
<td>Strategy for selecting optimal K. One of <code>majority_vote</code>, <code>gev</code>, <code>cv</code>, <code>db</code>, <code>kl</code>, <code>sil</code>, <code>dunn</code>, <code>ch</code>, <code>gap</code>, <code>aic</code>, <code>bic</code>. Unrecognized values fall back to <code>majority_vote</code>.</td>
</tr>
<tr>
<td><code>stopping_threshold</code></td>
<td>Integer</td>
<td><code>10</code></td>
<td>GEV elbow gain threshold as a percent (1-100): the minimum relative GEV increase that justifies adding another cluster. Used only when <code>stopping_mode = gev</code>.</td>
</tr>
<tr>
<td><code>data_percentage</code></td>
<td>Integer</td>
<td><code>100</code></td>
<td>Percentage of data used for clustering. <code>100</code> uses every timepoint; lower values draw a random subset.</td>
</tr>
<tr>
<td><code>initializer</code></td>
<td>String</td>
<td><code>Random</code></td>
<td>Cluster initialization: <code>Random</code> or <code>K-Means++</code>.</td>
</tr>
<tr>
<td><code>clustering_method</code></td>
<td>String</td>
<td><code>Modified K-Means Clustering</code></td>
<td>Clustering algorithm to use.</td>
</tr>
<tr>
<td><code>similarity_metric</code></td>
<td>String</td>
<td><code>Spatial Correlation</code></td>
<td>Map comparison metric: <code>Spatial Correlation</code> or <code>Cosine Similarity</code>. Used only by the spatial-similarity clustering method.</td>
</tr>
<tr>
<td><code>max_iterations</code></td>
<td>Integer</td>
<td><code>500</code></td>
<td>Maximum iterations per clustering run.</td>
</tr>
<tr>
<td><code>clustering_tolerance</code></td>
<td>Float</td>
<td><code>1e-6</code></td>
<td>Convergence tolerance threshold.</td>
</tr>
<tr>
<td><code>n_repeats</code></td>
<td>Integer</td>
<td><code>5</code></td>
<td>Number of clustering repetitions.</td>
</tr>
</tbody>
</table>

### Stopping Mode Options

| Value | Description |
|:------|:------------|
| `majority_vote` | Consensus across all validation criteria |
| `gev` | Global Explained Variance |
| `cv` | Cross-Validation Criterion |
| `db` | Davies-Bouldin Index |
| `kl` | Krzanowski-Lai Criterion |
| `sil` | Silhouette Score |
| `dunn` | Dunn Index |
| `ch` | Calinski-Harabasz Index |
| `gap` | Gap Statistic |
| `aic` | Akaike Information Criterion |
| `bic` | Bayesian Information Criterion |

### Clustering Method Options

| Value | Description |
|:------|:------------|
| `Modified K-Means Clustering` | Polarity-invariant K-means (recommended) |
| `Modified K-Means Clustering with Spatial Similarity` | Polarity-invariant K-means using the configured `similarity_metric` |
| `Topographic Atomize and Agglomerate Hierarchical Clustering` | TAAHC algorithm |

### Recommended Settings by Goal

| Goal | Repeats | Tolerance | Max Iterations |
|:-----|:--------|:----------|:---------------|
| Quick exploration | 5 | 1e-5 | 300 |
| Standard analysis | 20 | 1e-6 | 500 |
| Publication quality | 50-100 | 1e-7 | 1000 |

---

## Backfitting Configuration

### `[backfitting_config]`

Parameters for template assignment and segment refinement.

<table class="param-table">
<thead>
<tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr>
</thead>
<tbody>
<tr>
<td><code>backfit_to</code></td>
<td>String</td>
<td><code>all</code></td>
<td>Timepoints to backfit: <code>all</code> or <code>peaks</code>.</td>
</tr>
<tr>
<td><code>identify_short_window</code></td>
<td>Boolean</td>
<td><code>False</code></td>
<td>Automatically optimize minimum segment duration.</td>
</tr>
<tr>
<td><code>filter_segments</code></td>
<td>Boolean</td>
<td><code>False</code></td>
<td>Apply segment refinement to short segments.</td>
</tr>
<tr>
<td><code>filter_segments_less_than</code></td>
<td>Integer</td>
<td><code>20</code></td>
<td>Minimum segment duration threshold (milliseconds).</td>
</tr>
<tr>
<td><code>filter_segments_option</code></td>
<td>String</td>
<td><code>smooth</code></td>
<td>Strategy for handling short segments.</td>
</tr>
<tr>
<td><code>convergence_epsilon</code></td>
<td>Float</td>
<td><code>1e-6</code></td>
<td>Convergence criterion for smoothing algorithm.</td>
</tr>
<tr>
<td><code>half_window_size</code></td>
<td>Integer</td>
<td><code>3</code></td>
<td>Window size parameter for smoothing (samples).</td>
</tr>
<tr>
<td><code>smoothness_penalty</code></td>
<td>Float</td>
<td><code>5</code></td>
<td>Non-smoothness penalty (lambda) for smoothing.</td>
</tr>
</tbody>
</table>

### Segment Handling Options

| Value | Description | Data Preserved |
|:------|:------------|:---------------|
| `remove` | Delete short segments, create gaps | No |
| `replace_high` | Replace with dominant neighbor | Yes |
| `replace_half` | Split between neighbors | Yes |
| `smooth` | Temporal smoothing algorithm | Yes |

### Typical Duration Thresholds

| Threshold | Strictness | Effect |
|:----------|:-----------|:-------|
| 10 ms | Lenient | Keep most segments |
| 20 ms | Moderate | Standard choice |
| 30 ms | Strict | Remove more noise |
| 40 ms | Very strict | Only stable segments |

---

## Feature Extraction Configuration

### `[features_config]`

Parameters for microstate metric computation.

<table class="param-table">
<thead>
<tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr>
</thead>
<tbody>
<tr>
<td><code>export_format</code></td>
<td>String</td>
<td><code>.csv</code></td>
<td>Output file format.</td>
</tr>
<tr>
<td><code>feature_list</code></td>
<td>String</td>
<td><code>OCC, DUR, COV</code></td>
<td>Comma-separated list of features to extract.</td>
</tr>
<tr>
<td><code>feature_mode</code></td>
<td>String</td>
<td><code>averaged</code></td>
<td>Analysis mode: <code>averaged</code>, <code>sliding</code>, or <code>pre_post</code>.</td>
</tr>
<tr>
<td><code>feature_types</code></td>
<td>String</td>
<td><code>real</code></td>
<td>Comparison types to compute: <code>real</code>, <code>surrogate</code> and/or <code>random</code>.</td>
</tr>
<tr>
<td><code>sliding_window_size</code></td>
<td>Float</td>
<td><code>1</code></td>
<td>Window duration in seconds (for sliding mode).</td>
</tr>
<tr>
<td><code>duration_method</code></td>
<td>String</td>
<td><code>geometric</code></td>
<td>How per-segment microstate run lengths are summarised into the DUR feature. See options below.</td>
</tr>
</tbody>
</table>

### Duration Aggregation Options

The `duration_method` parameter controls how the per-segment run lengths of each microstate are summarised into the reported mean duration (DUR / MMD).

| Value | Description | When to use |
|:------|:------------|:------------|
| `geometric` *(default)* | Geometric mean of run lengths × `1000/fs`. | Default; robust to long-tail outliers that inflate the arithmetic mean for high-coverage microstates. |
| `arithmetic` | Mean of run lengths with the `(N-1)/fs` interval convention. | When DUR must be algebraically consistent with COV and OCC (`COV ≈ DUR × OCC`). |
| `median` | Median of run lengths × `1000/fs`. | Robust central-tendency reporting. |
| `trimmed_mean` | 10% symmetric trimmed mean × `1000/fs` (falls back to mean if fewer than 11 segments). | Robust alternative when occasional very long segments distort the mean. |

{: .note }
> The default switched from `arithmetic` to `geometric` to better reflect the typical persistence of dominant microstates, whose run-length distributions are heavy-tailed. Specify `duration_method = arithmetic` to reproduce results from earlier versions or to keep the COV / DUR / OCC identity exact.

### Export Format Options

| Value | Description | Best For |
|:------|:------------|:---------|
| `.csv` | Comma-separated values | Excel, R, general use |
| `.pkl` | Python pickle | Python workflows |
| `.hdf` | HDF5 format | Large datasets |
| `.json` | JSON format | Web, APIs |

### Feature Codes

| Code | Feature | Category |
|:-----|:--------|:---------|
| `COV` | Coverage | Classical |
| `OCC` | Occurrence | Classical |
| `MMD` / `DUR` | Mean Duration | Classical |
| `GEV` | Global Explained Variance | Classical |
| `TP` | Transition Probabilities | Sequential |
| `LZC` | Lempel-Ziv Complexity | Complexity |
| `ER` | Entropy Rate | Complexity |
| `HE` | Hurst Exponent | Complexity |
| `ERR` | Entropy Representation Ratio | Complexity |

### Feature Mode Options

| Value | Description | Output |
|:------|:------------|:-------|
| `averaged` | Whole-recording average | One value per recording |
| `sliding` | Sliding window analysis | Time series of values |
| `pre_post` | Trial-level pre/post-event extraction | Per-trial values |

---

## Source Localization Configuration

### `[source_config]`

Parameters for cortical source estimation.

<table class="param-table">
<thead>
<tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr>
</thead>
<tbody>
<tr>
<td><code>inverse_method</code></td>
<td>String</td>
<td><code>dSPM</code></td>
<td>Inverse solution algorithm.</td>
</tr>
<tr>
<td><code>source_localization_method</code></td>
<td>String</td>
<td><code>tess</code></td>
<td>Source reconstruction approach.</td>
</tr>
<tr>
<td><code>n_permutations</code></td>
<td>Integer</td>
<td><code>2000</code></td>
<td>Number of permutations for TESS method.</td>
</tr>
<tr>
<td><code>spacing</code></td>
<td>String</td>
<td><code>ico3</code></td>
<td>Source space resolution.</td>
</tr>
<tr>
<td><code>anatomy_subjects_dir</code></td>
<td>Path</td>
<td><code>[]</code></td>
<td>Path to FreeSurfer subjects directory. Empty for template.</td>
</tr>
</tbody>
</table>

### Inverse Method Options

| Value | Full Name | Description |
|:------|:----------|:------------|
| `MNE` | Minimum Norm Estimate | Basic distributed source |
| `dSPM` | Dynamic SPM | Noise-normalized (recommended) |
| `sLORETA` | Standardized LORETA | Current-normalized |
| `eLORETA` | Exact LORETA | Exact zero localization error |

### Source Space Resolution

| Value | Approximate Sources | Use Case |
|:------|:--------------------|:---------|
| `ico3` | ~1,280 | Quick analysis, exploration |
| `ico4` | ~5,120 | Publication quality |
| `ico5` | ~20,480 | High-resolution (slow) |

### Reconstruction Method Options

| Value | Description |
|:------|:------------|
| `avg` | Temporal averaging by microstate |
| `tess` | TESS two-stage GLM approach |

---

## Quick Reference by Analysis Type

### Resting-State Analysis

```ini
[preprocessing_config]
filter_method = fir
lowcut_freq = 2
highcut_freq = 20

[clustering_config]
n_maps = 4
clustering_method = Modified K-Means Clustering
n_repeats = 20

[backfitting_config]
filter_segments_option = remove

[features_config]
feature_mode = averaged
feature_list = COV,OCC,MMD,GEV,TP
```

### Event-Related Analysis

```ini
[preprocessing_config]
filter_method = fir
lowcut_freq = 1
highcut_freq = 40
downsample_data = False

[clustering_config]
data_percentage = 100
clustering_method = Topographic Atomize and Agglomerate Hierarchical Clustering

[backfitting_config]
filter_segments_option = smooth

[features_config]
feature_mode = pre_post
```

### Clinical/Group Comparison

```ini
[clustering_config]
n_maps = auto
stopping_mode = majority_vote
n_repeats = 50

[features_config]
feature_list = COV,OCC,MMD,GEV,TP,LZC
export_format = .csv
```

---

## Parameter Validation

EEG-COMET validates parameters on loading:

| Validation | Check |
|:-----------|:------|
| Required fields | All mandatory parameters present |
| Type checking | Values match expected types |
| Range checking | Values within valid ranges |
| Path validation | Directories exist and are accessible |
| Consistency | Related parameters are compatible |

### Common Validation Errors

| Error | Cause | Solution |
|:------|:------|:---------|
| Missing required parameter | Field not specified | Add to config file |
| Invalid path | Directory doesn't exist | Create directory or fix path |
| Out of range | Value outside limits | Use value within valid range |
| Type mismatch | Wrong data type | Check expected format |
| `highcut_freq` at or above Nyquist | Low-pass cutoff too close to the recording or target sampling rate | Lower `highcut_freq`, or raise `sampling_rate` above twice `highcut_freq` |
| `lowcut_freq` not below `highcut_freq` | Passband bounds inverted or equal | Set `lowcut_freq` below `highcut_freq` |

---

## Environment Variables

Some parameters can be overridden via environment variables:

| Variable | Overrides |
|:---------|:----------|
| `EEG_COMET_DATA_DIR` | `input_folder` |
| `EEG_COMET_OUTPUT_DIR` | `output_folder` |
| `SUBJECTS_DIR` | `anatomy_subjects_dir` |

---

## See Also

- [Getting Started]({% link getting-started.md %}) - Tutorial using these parameters
- [Module Documentation]({% link modules/index.md %}) - Detailed parameter explanations
- [Installation]({% link installation.md %}) - Setting up EEG-COMET

