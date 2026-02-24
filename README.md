# poseToRecord

Convert [MMPose](https://github.com/open-mmlab/mmpose) COCO-style pose estimation JSON output into [**movement**](https://movement.neuroinformatics.dev/)-format NetCDF records and compute a rich set of dyadic quality metrics for social interaction analysis.

---

## What it does

1. **Converts** MMPose skeleton JSON → `xarray.Dataset` in the `movement` format (`tracking.nc`)
2. **Cleans** the data: confidence filtering, dyadic presence filtering, continuity segmentation
3. **Computes metrics**: speed, acceleration, kinetic energy, interpersonal distance, facingness, congruent motion
4. **Generates outputs**: CSVs, publication-quality plots, and a self-contained HTML report

Designed for dyadic social interaction recordings (e.g., child–clinician ADOS assessments).

---

## Installation

### Requirements

- Python ≥ 3.10
- [movement](https://movement.neuroinformatics.dev/) library (for the xarray format)
- Standard scientific stack

### Install dependencies

```bash
pip install movement xarray numpy pandas matplotlib scipy
```

Or with conda:

```bash
conda install -c conda-forge movement xarray numpy pandas matplotlib scipy
```

### Install the module

The module is self-contained. Clone/copy the `poseToRecord/` directory into your project and make it importable:

```bash
# From the directory containing poseToRecord/
pip install -e .
# or just add the parent directory to your PYTHONPATH
```

---

## Quick start

### Command line

```bash
python -m poseToRecord.pipeline \
    --input   results_skeleton_8090_T2a_ADOS.json \
    --output  output_records/ \
    --identity-map "0:child,1:clinician,2:parent" \
    --dyadic-individuals child clinician \
    --fps 20 \
    --conf-threshold 0.3 \
    --min-valid-kp 5 \
    --min-segment-frames 300 \
    --log-level INFO
```

### Python API

```python
from poseToRecord import run_pipeline

summary = run_pipeline(
    input_json="results_skeleton_8090_T2a_ADOS.json",
    output_dir="output_records/",
    identity_map={0: "child", 1: "clinician", 2: "parent"},
    dyadic_individuals=["child", "clinician"],
    fps=20.0,
    conf_threshold=0.3,
    min_valid_kp=5,
    min_segment_frames=300,
)
print(summary)
# {'video_name': '...', 'segments_kept': 27, 'total_valid_frames': 12453, ...}
```

### Load output with movement

```python
from movement.io import load_poses

ds = load_poses.from_file("output_records/video_name/tracking.nc", source_software="LightningPose")
```

---

## Input format

MMPose JSON skeleton output:

```json
{
  "meta_info": {
    "keypoint_id2name": {"0": "nose", "1": "left_eye", ...}
  },
  "instance_info": [
    {
      "frame_id": 0,
      "instances": [
        {
          "keypoints": [[x, y], ...],
          "keypoint_scores": [0.95, 0.87, ...],
          "keypoints_label": 1
        }
      ]
    }
  ]
}
```

**`keypoints_label`** is the integer tracking ID. You map these to individual names via `--identity-map`.

---

## Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `identity_map` | `--identity-map` | required | Mapping from tracking ID to name, e.g. `"0:child,1:clinician,2:parent"` |
| `dyadic_individuals` | `--dyadic-individuals` | first two in map | The two individuals that must co-occur for a frame to be valid |
| `fps` | `--fps` | `25.0` | Frames per second of the source video |
| `conf_threshold` | `--conf-threshold` | `0.3` | Confidence score below which keypoints are zeroed |
| `min_valid_kp` | `--min-valid-kp` | `5` | Min keypoints above threshold per individual per frame |
| `min_segment_frames` | `--min-segment-frames` | `300` | Min consecutive valid frames to keep a segment (~15 s at 20 fps) |
| `congruent_window` | `--congruent-window` | `60` | Rolling window (frames) for congruent-motion correlation |
| — | `--no-metrics` | off | Skip metric computation (convert + filter only) |
| — | `--no-report` | off | Skip HTML report (still writes CSVs and plots) |
| — | `--log-level` | `INFO` | Console log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Output structure

```
output_dir/
└── {video_name}/
    ├── tracking.nc                  ← Full record (movement format)
    ├── pipeline.log                 ← Full pipeline log
    ├── cleaning_stats.csv           ← Frames dropped per cleaning step
    ├── kp_coverage_before.csv       ← Keypoint visibility before cleaning
    ├── kp_coverage_after.csv        ← Keypoint visibility in kept segments
    ├── video_metrics_raw_2d.csv     ← All segments concatenated (raw units)
    ├── video_metrics_normalised.csv ← All segments (trunk-height normalised)
    ├── video_metrics_summary.csv    ← Per-segment + overall summary stats
    ├── plots/
    │   ├── video_speed_centroid.png
    │   ├── video_speed_centroid_dist.png  ← Distribution across segments
    │   ├── video_kp_comparison_diff.png   ← Centroid − trunk speed difference
    │   └── ... (one overlay + one distribution plot per metric)
    ├── report_{video_name}.html     ← Self-contained HTML report
    └── segments/
        ├── seg_001/
        │   ├── tracking.nc
        │   ├── metrics_raw_2d.csv
        │   ├── metrics_normalised.csv
        │   ├── metrics_summary.csv
        │   └── plots/
        ├── seg_002/
        └── ...
```

---

## Metrics

All metrics are computed in two versions: **raw** (pixel units) and **normalised** (divided by trunk height).

| Metric | Description |
|--------|-------------|
| **Centroid speed** | Displacement of the body centroid between consecutive frames, computed over the intersection of visible keypoints at t and t−1 |
| **Trunk speed** | Same as centroid speed but restricted to the 4 trunk keypoints (left/right shoulder + left/right hip) |
| **Kinetic energy** | Sum of squared keypoint displacements: `KE(t) = Σ ‖pos_kp(t) − pos_kp(t-1)‖²` |
| **Acceleration** | Frame-to-frame change in centroid speed |
| **Interpersonal distance** | Euclidean distance between the two individuals' centroids |
| **Approach / retreat** | Rate of change of interpersonal distance (negative = approaching) |
| **Facingness** | Cosine similarity of torso heading vectors (hips→shoulders): +1 = same direction, −1 = face-to-face |
| **Congruent motion** | Rolling-window Pearson correlation of the two individuals' speed time series |
| **Global agitation** | Mean kinetic energy across both dyadic individuals |

### Design notes

- **Missing keypoints** are encoded as `[0.0, 0.0]` (MMPose convention). They are excluded from all computations.
- **Centroid speed** uses the *intersection* of visible keypoints at frames t and t−1 to avoid centroid-shift artifacts when a keypoint appears/disappears. Transitions with fewer than 3 shared keypoints produce `NaN`.
- **Trunk height** is `AVG(d(left_shoulder, left_hip), d(right_shoulder, right_hip))` with single-side fallback, smoothed by a rolling median (window = 25 frames).
- The `kp_set_changed` flag in the output CSVs marks frames where the visible keypoint composition changed.

---

## Module structure

```
poseToRecord/
├── __init__.py    — Public API
├── convert.py     — JSON → xarray.Dataset
├── filter.py      — Confidence, dyadic, continuity filtering + CleaningStats
├── metrics.py     — All kinematic and dyadic quality metrics
├── report.py      — CSV export, matplotlib plots, HTML report
├── io.py          — Record dataclass + dump_records (movement-compatible)
└── pipeline.py    — Orchestration + CLI entry point
```

---

## Cleaning pipeline

Frames are filtered in four successive stages; each stage is logged and reported in `cleaning_stats.csv`:

1. **Negative coordinate clipping** — `position.clip(min=0.0)`
2. **Confidence filtering** — keypoints with `score < conf_threshold` are zeroed
3. **Dyadic presence filter** — frames where either dyadic individual has fewer than `min_valid_kp` above-threshold keypoints are excluded
4. **Continuity filter** — contiguous valid segments shorter than `min_segment_frames` are discarded

### On keypoint coverage "before" vs "after"

**Before** = coverage over all frames where the individual appears in the full raw recording.
**After** = coverage over only the kept segments.

Coverage can legitimately be *higher* after cleaning: the kept segments are a high-quality biased sample (specifically selected because both individuals were reliably detected). This is expected and desirable — the cleaning isolated the well-detected periods. No smoothing is applied to keypoint presence.

---

## Programmatic lower-level access

```python
from poseToRecord import convert_json_to_dataset, apply_cleaning_pipeline, compute_all_metrics

# Step 1: convert
raw_ds = convert_json_to_dataset("skeleton.json", {0: "child", 1: "clinician"}, fps=20.0)

# Step 2: filter
full_ds, segments, stats = apply_cleaning_pipeline(
    raw_ds,
    dyadic_individuals=["child", "clinician"],
    conf_threshold=0.3,
    min_valid_kp=5,
    min_segment_frames=300,
)

# Step 3: metrics for one segment
metric_dfs = compute_all_metrics(segments[0], dyadic_individuals=["child", "clinician"])
raw_df  = metric_dfs["raw"]         # per-frame, pixel units
norm_df = metric_dfs["normalised"]  # per-frame, trunk-height normalised
```

---

## Keypoints

17 COCO keypoints (standard MMPose output):

| Index | Name | Index | Name |
|-------|------|-------|------|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

---

## License

See the repository root for license information.
