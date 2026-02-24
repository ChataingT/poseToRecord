"""Report generation: CSV export, matplotlib plots, and HTML report.

This module produces three types of output for each processed video:

1. **CSV files** — per-frame metrics (raw 2D and trunk-height normalised) at
   the segment level, plus aggregated summaries at the video level.
2. **Matplotlib plots** — publication-quality time-series and summary figures,
   saved as PNG and embedded in the HTML report.
3. **HTML report** — a self-contained document with metric definitions,
   interpretation tooltips, embedded plots, and summary tables.

Two-level output structure
--------------------------
::

    output_dir/{video_name}/
    ├── tracking.nc                 ← full record
    ├── cleaning_stats.csv
    ├── kp_coverage_before.csv
    ├── kp_coverage_after.csv
    ├── video_metrics_raw_2d.csv    ← all segments concatenated (+ segment_id col)
    ├── video_metrics_summary.csv   ← one row per segment + overall row
    ├── plots/                      ← video-level overlay plots
    │   └── *.png
    ├── report_{video_name}.html    ← self-contained HTML report
    └── segments/
        └── seg_{NNN}/
            ├── tracking.nc
            ├── metrics_raw_2d.csv
            ├── metrics_normalised.csv
            ├── metrics_summary.csv
            └── plots/
                └── *.png
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .filter import CleaningStats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

PLOT_DPI = 150
FIGSIZE_WIDE = (14, 4)
FIGSIZE_TALL = (14, 6)

_PALETTE = {
    "child":     "#2196F3",   # blue
    "clinician": "#F44336",   # red
    "parent":    "#4CAF50",   # green
    "global":    "#9C27B0",   # purple
}

def _color(name: str) -> str:
    for key, col in _PALETTE.items():
        if key in name.lower():
            return col
    return "#607D8B"


# ---------------------------------------------------------------------------
# Metric metadata (for report text)
# ---------------------------------------------------------------------------

METRIC_META: dict[str, dict] = {
    "speed_centroid": {
        "title": "Centroid Speed",
        "unit": "px/frame (raw) | trunk-height/frame (normalised)",
        "definition": (
            "Euclidean distance of the body centroid between consecutive frames, "
            "computed over the intersection of visible keypoints at times t and t−1."
        ),
        "interpretation": (
            "Higher values indicate faster overall movement. "
            "Frames where fewer than 3 keypoints are shared with the previous frame are NaN. "
            "Check the 'kp_set_changed' column to identify frames where the visible keypoint "
            "set changed — these may inflate or deflate the apparent speed."
        ),
    },
    "speed_trunk": {
        "title": "Trunk Speed",
        "unit": "px/frame (raw) | trunk-height/frame (normalised)",
        "definition": (
            "Speed computed using only the four trunk keypoints "
            "(left/right shoulder and left/right hip)."
        ),
        "interpretation": (
            "Trunk speed is more stable than whole-body centroid speed because the trunk "
            "keypoints are larger and usually more reliably detected. "
            "Comparing trunk speed with centroid speed helps detect artifacts caused by "
            "missing extremity keypoints."
        ),
    },
    "kinetic_energy": {
        "title": "Kinetic Energy (Agitation)",
        "unit": "px² (raw) | trunk-height² (normalised)",
        "definition": (
            "Sum of squared displacements over all visible keypoints shared between "
            "consecutive frames: KE(t) = Σ ‖pos_kp(t) − pos_kp(t-1)‖². "
            "Proportional to kinetic energy under a uniform mass assumption."
        ),
        "interpretation": (
            "Unlike centroid speed, kinetic energy captures distributed body movement — "
            "e.g., waving an arm while the body stays still still produces high KE. "
            "It is more robust to centroid-shift artifacts from missing keypoints. "
            "High values indicate high overall agitation."
        ),
    },
    "acceleration_centroid": {
        "title": "Centroid Acceleration",
        "unit": "px/frame² (raw)",
        "definition": "Absolute frame-to-frame change in centroid speed.",
        "interpretation": (
            "High acceleration indicates sudden starts or stops. "
            "It is more sensitive to noise than speed — "
            "consider smoothing before drawing conclusions."
        ),
    },
    "interpersonal_distance_centroid": {
        "title": "Interpersonal Distance (Centroid)",
        "unit": "px (raw) | trunk-height (normalised)",
        "definition": (
            "Euclidean distance between the centroids of the two dyadic individuals, "
            "computed when both have at least 3 visible keypoints."
        ),
        "interpretation": (
            "Low values indicate physical proximity. "
            "Normalised by mean trunk height, the value approximates the number of "
            "'body lengths' separating the two individuals."
        ),
    },
    "interpersonal_approach": {
        "title": "Approach / Retreat",
        "unit": "px/frame (raw)",
        "definition": "Frame-to-frame change in interpersonal distance.",
        "interpretation": (
            "Negative values mean the individuals are approaching each other; "
            "positive values mean they are moving apart."
        ),
    },
    "facingness": {
        "title": "Facingness",
        "unit": "cosine similarity [-1, +1]",
        "definition": (
            "Cosine similarity of the two individuals' torso heading vectors. "
            "Heading = mid(left_shoulder, right_shoulder) − mid(left_hip, right_hip), "
            "pointing 'upward' from hips toward shoulders. "
            "Score: +1 = same direction (side-by-side), −1 = face-to-face, 0 = perpendicular."
        ),
        "interpretation": (
            "A score near −1 suggests the two individuals are facing each other "
            "(front-to-front engagement). A score near +1 means they are oriented the same way "
            "(e.g. both watching a screen). "
            "Note: this is a 2D projection of torso orientation; camera angle affects the reading."
        ),
    },
    "congruent_motion": {
        "title": "Congruent Motion (Synchrony)",
        "unit": "Pearson r [-1, +1]",
        "definition": (
            "Rolling-window Pearson correlation of the two individuals' centroid speed "
            "time series. Measures whether both individuals tend to move (or stay still) "
            "at the same times."
        ),
        "interpretation": (
            "High positive values (near +1) indicate behavioural synchrony or motor mimicry — "
            "the two individuals move in temporal concert. "
            "Values near 0 indicate independent movement patterns. "
            "Negative values (rare) suggest alternating activity. "
            "Windows with more than 30% NaN speed values are excluded."
        ),
    },
    "agitation_global_ke": {
        "title": "Global Agitation",
        "unit": "px² (raw)",
        "definition": "Mean kinetic energy across all dyadic individuals at each frame.",
        "interpretation": (
            "A single-number summary of how much movement is happening in the interaction "
            "at each moment, regardless of who is moving."
        ),
    },
}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def save_segment_csvs(
    seg_dir: Path,
    raw_df: pd.DataFrame,
    norm_df: pd.DataFrame,
) -> None:
    """Save per-frame and summary CSVs for a single segment.

    Parameters
    ----------
    seg_dir : Path
        Directory for this segment's outputs (created if absent).
    raw_df : pd.DataFrame
        Per-frame metrics in pixel / original units.
    norm_df : pd.DataFrame
        Per-frame metrics normalised by trunk height.
    """
    seg_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(seg_dir / "metrics_raw_2d.csv")
    norm_df.to_csv(seg_dir / "metrics_normalised.csv")
    # Summary stats
    summary = pd.concat(
        [raw_df.describe().T.add_prefix("raw_"), norm_df.describe().T.add_prefix("norm_")],
        axis=1,
    )
    # Align pct_valid with the index (describe() may skip non-numeric cols)
    pct_valid = raw_df.notna().mean() * 100
    summary["pct_valid_raw"] = pct_valid.reindex(summary.index)
    summary.to_csv(seg_dir / "metrics_summary.csv")
    logger.info("Segment CSVs saved to %s", seg_dir)


def save_video_csvs(
    video_dir: Path,
    all_raw: list[pd.DataFrame],
    all_norm: list[pd.DataFrame],
    stats: CleaningStats,
) -> None:
    """Save video-level aggregated CSVs and cleaning statistics.

    Parameters
    ----------
    video_dir : Path
    all_raw : list of per-segment raw DataFrames
    all_norm : list of per-segment normalised DataFrames
    stats : CleaningStats
    """
    video_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate with segment_id column
    tagged_raw, tagged_norm = [], []
    for i, (raw, norm) in enumerate(zip(all_raw, all_norm)):
        r = raw.copy(); r["segment_id"] = i
        n = norm.copy(); n["segment_id"] = i
        tagged_raw.append(r)
        tagged_norm.append(n)

    if tagged_raw:
        video_raw = pd.concat(tagged_raw, ignore_index=True)
        video_raw.to_csv(video_dir / "video_metrics_raw_2d.csv")
        video_norm = pd.concat(tagged_norm, ignore_index=True)
        video_norm.to_csv(video_dir / "video_metrics_normalised.csv")

        # Per-segment summary + overall row
        rows = []
        for i, raw in enumerate(all_raw):
            row = raw.describe().loc[["mean", "std", "50%"]].T
            row.columns = ["mean", "std", "median"]
            row["segment_id"] = i
            row["n_frames"] = len(raw)
            rows.append(row)
        # Overall
        overall = video_raw.drop(columns=["segment_id"]).describe().loc[["mean", "std", "50%"]].T
        overall.columns = ["mean", "std", "median"]
        overall["segment_id"] = "overall"
        overall["n_frames"] = len(video_raw)
        rows.append(overall)
        summary_df = pd.concat(rows)
        summary_df.to_csv(video_dir / "video_metrics_summary.csv")

    # Cleaning stats
    _save_cleaning_stats(video_dir, stats)
    logger.info("Video-level CSVs saved to %s", video_dir)


def _save_cleaning_stats(video_dir: Path, stats: CleaningStats) -> None:
    """Write cleaning statistics CSVs."""
    fps = 20.0  # default; will be overridden if passed

    step_rows = [
        {
            "step": "original",
            "frames": stats.total_frames,
            "frames_dropped": 0,
            "pct_dropped": 0.0,
        },
        {
            "step": "after_confidence_filter",
            "frames": stats.frames_after_conf,
            "frames_dropped": stats.total_frames - stats.frames_after_conf,
            "pct_dropped": 100.0 * (stats.total_frames - stats.frames_after_conf) / max(stats.total_frames, 1),
        },
        {
            "step": "after_dyadic_filter",
            "frames": stats.frames_after_dyadic,
            "frames_dropped": stats.total_frames - stats.frames_after_dyadic,
            "pct_dropped": 100.0 * (stats.total_frames - stats.frames_after_dyadic) / max(stats.total_frames, 1),
        },
        {
            "step": "after_continuity_filter",
            "frames": stats.frames_after_continuity,
            "frames_dropped": stats.total_frames - stats.frames_after_continuity,
            "pct_dropped": 100.0 * (stats.total_frames - stats.frames_after_continuity) / max(stats.total_frames, 1),
        },
    ]
    pd.DataFrame(step_rows).to_csv(video_dir / "cleaning_stats.csv", index=False)

    # Keypoint coverage tables
    def _cov_to_df(cov: dict) -> pd.DataFrame:
        rows = []
        for ind, kp_dict in cov.items():
            for kp, pct in kp_dict.items():
                rows.append({"individual": ind, "keypoint": kp, "coverage_pct": pct})
        return pd.DataFrame(rows)

    _cov_to_df(stats.kp_coverage_before).to_csv(video_dir / "kp_coverage_before.csv", index=False)
    _cov_to_df(stats.kp_coverage_after).to_csv(video_dir / "kp_coverage_after.csv", index=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_segment_plots(seg_dir: Path, raw_df: pd.DataFrame, norm_df: pd.DataFrame) -> list[Path]:
    """Generate and save per-segment time-series plots.

    Returns
    -------
    list[Path]
        Paths to all saved PNG files.
    """
    plot_dir = seg_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    saved += _plot_speed_panel(plot_dir, raw_df, norm_df, "segment")
    saved += _plot_acceleration_panel(plot_dir, raw_df, "segment")
    saved += _plot_ke_panel(plot_dir, raw_df, norm_df, "segment")
    saved += _plot_interpersonal(plot_dir, raw_df, norm_df, "segment")
    saved += _plot_facingness(plot_dir, raw_df, "segment")
    saved += _plot_congruent(plot_dir, raw_df, "segment")

    logger.info("Segment plots saved: %d files in %s", len(saved), plot_dir)
    return saved


def save_video_plots(
    video_dir: Path,
    all_raw: list[pd.DataFrame],
    all_norm: list[pd.DataFrame],
    stats: CleaningStats | None = None,
    fps: float = 25.0,
) -> list[Path]:
    """Generate video-level overlay, distribution, and comparison plots.

    Parameters
    ----------
    video_dir : Path
    all_raw : list of per-segment raw DataFrames
    all_norm : list of per-segment normalised DataFrames
    stats : CleaningStats or None
        When provided, generates the segment-length distribution plot.
    fps : float
        Frames per second, used to convert frame counts to seconds in the
        segment-length distribution plot.

    Returns
    -------
    list[Path]
        Paths to all saved PNG files.
    """
    plot_dir = video_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    if not all_raw:
        return saved

    # Concatenated data for overall trend (with segment_id)
    tagged = []
    for i, df in enumerate(all_raw):
        d = df.copy(); d["segment_id"] = i; d["x"] = np.arange(len(df))
        tagged.append(d)
    merged = pd.concat(tagged, ignore_index=True)

    # Overlay time-series plots (per-segment light lines + smoothed overall trend)
    saved += _plot_overlay(plot_dir, all_raw, all_norm, merged)

    # Centroid − trunk speed difference plots
    saved += _plot_kp_comparison(plot_dir, all_raw, merged)

    # Per-metric distribution plots (distribution of per-segment means & medians)
    for base_col in METRIC_META:
        path = _plot_metric_distribution(plot_dir, all_raw, base_col)
        if path is not None:
            saved.append(path)

    # Segment length distribution
    if stats is not None:
        path = _plot_segment_length_distribution(plot_dir, stats, fps)
        if path is not None:
            saved.append(path)

    logger.info("Video-level plots saved: %d files in %s", len(saved), plot_dir)
    return saved


def _plot_segment_length_distribution(
    plot_dir: Path,
    stats: CleaningStats,
    fps: float,
) -> Path | None:
    """Histogram + rugplot of segment durations in seconds.

    The histogram shows the frequency distribution of segment lengths.
    Short vertical rug marks at the bottom indicate each individual segment.
    Reference lines mark the mean, median, and the minimum-length threshold.

    Parameters
    ----------
    plot_dir : Path
    stats : CleaningStats
    fps : float

    Returns
    -------
    Path or None
        None if there are no segments to plot.
    """
    if not stats.segment_lengths:
        return None

    durations = np.array(stats.segment_lengths, dtype=float) / max(fps, 1.0)
    n = len(durations)

    fig, ax = plt.subplots(figsize=(10, 4))
    n_bins = min(20, max(5, n // 2))
    ax.hist(durations, bins=n_bins, color="#1565C0", alpha=0.55,
            edgecolor="white", linewidth=0.5)

    # Rug plot — tiny vertical tick marks for each segment
    for d in durations:
        ax.axvline(d, color="#0D47A1", alpha=0.45, linewidth=0.7, ymax=0.06)

    mean_dur   = float(np.mean(durations))
    median_dur = float(np.median(durations))
    min_thresh = stats.min_segment_frames / max(fps, 1.0)

    ax.axvline(mean_dur,   color="#E53935", linewidth=1.8, linestyle="--",
               label=f"Mean: {mean_dur:.1f} s")
    ax.axvline(median_dur, color="#FB8C00", linewidth=1.8, linestyle=":",
               label=f"Median: {median_dur:.1f} s")
    ax.axvline(min_thresh, color="#757575", linewidth=1.2, linestyle="-.",
               label=f"Min threshold: {min_thresh:.1f} s")

    ax.set_xlabel("Segment duration (seconds)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Segment Length Distribution  (N = {n} segments kept)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.98, 0.95,
        f"Min: {durations.min():.1f} s   Max: {durations.max():.1f} s",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555",
    )
    fig.tight_layout()
    path = plot_dir / "segment_length_dist.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Segment length distribution plot saved: %s", path)
    return path


def _plot_metric_distribution(
    plot_dir: Path,
    all_raw: list[pd.DataFrame],
    base_col: str,
) -> Path | None:
    """Strip + box plot: distribution of per-segment means and medians.

    For each column matching ``base_col`` (e.g. ``child_speed_centroid``,
    ``clinician_speed_centroid``), two subplots are shown side by side:

    * **Left** — distribution of per-segment *means*: one dot per segment.
    * **Right** — distribution of per-segment *medians*: one dot per segment.

    The box marks the IQR; the dashed horizontal line shows the global value
    computed across all frames in all segments.

    Per-keypoint columns (``_speed_kp_*``) and boolean flags are excluded.

    Parameters
    ----------
    plot_dir : Path
    all_raw : list of per-segment DataFrames
    base_col : str
        Key in ``METRIC_META`` (e.g. ``"speed_centroid"``).

    Returns
    -------
    Path or None
        None if no matching columns are found.
    """
    # Collect matching columns (exclude per-keypoint and boolean-flag columns)
    sample_cols: list[str] = []
    for df in all_raw:
        for c in df.columns:
            if (
                base_col in c
                and c not in sample_cols
                and not c.endswith("_changed")
                and "_speed_kp_" not in c
            ):
                sample_cols.append(c)

    if not sample_cols:
        return None

    n_segs = len(all_raw)
    rng = np.random.RandomState(0)

    # Compute per-segment means, medians, and aggregate all values
    seg_means:   dict[str, list[float]] = {c: [] for c in sample_cols}
    seg_medians: dict[str, list[float]] = {c: [] for c in sample_cols}
    all_vals:    dict[str, list[float]] = {c: [] for c in sample_cols}

    for df in all_raw:
        for c in sample_cols:
            vals = df[c].dropna() if c in df.columns else pd.Series(dtype=float)
            seg_means[c].append(float(vals.mean())   if len(vals) > 0 else np.nan)
            seg_medians[c].append(float(vals.median()) if len(vals) > 0 else np.nan)
            all_vals[c].extend(vals.tolist())

    n_cols = len(sample_cols)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, n_cols * 2.8 + 2), 5))

    for ax, stat_dict, stat_label in zip(
        axes,
        [seg_means, seg_medians],
        ["Mean per segment", "Median per segment"],
    ):
        for pos_idx, c in enumerate(sample_cols, start=1):
            vals = np.array(stat_dict[c], dtype=float)
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                continue

            ind = c.split(f"_{base_col}")[0] if f"_{base_col}" in c else c
            color = _color(ind)

            # Box plot (IQR, no fliers — individual dots shown separately)
            ax.boxplot(
                [valid],
                positions=[pos_idx],
                widths=0.35,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.35),
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(linewidth=1.0),
                capprops=dict(linewidth=1.0),
            )

            # Jittered dots — one per segment
            jitter = rng.uniform(-0.12, 0.12, len(valid))
            ax.scatter(pos_idx + jitter, valid, color=color, alpha=0.75, s=28, zorder=3)

            # Global reference line (dashed) — mean-of-all-frames or median-of-all-frames
            gv = np.array(all_vals[c], dtype=float)
            global_ref = (
                float(np.nanmean(gv))   if stat_label.startswith("Mean")
                else float(np.nanmedian(gv))
            )
            ax.hlines(
                global_ref, pos_idx - 0.25, pos_idx + 0.25,
                colors=color, linewidths=1.4, linestyles="--", alpha=0.7,
                label=f"Global {stat_label.split()[0].lower()} ({ind}): {global_ref:.3g}",
            )

        short_labels = [
            c.split(f"_{base_col}")[0] if f"_{base_col}" in c else c
            for c in sample_cols
        ]
        ax.set_xticks(range(1, n_cols + 1))
        ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(stat_label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        unit_str = METRIC_META.get(base_col, {}).get("unit", "")
        ax.set_ylabel(unit_str, fontsize=8)
        if n_cols <= 4:
            ax.legend(fontsize=7, loc="upper right")

    meta = METRIC_META.get(base_col, {})
    fig.suptitle(
        f"{meta.get('title', base_col)} — per-segment distribution  (N = {n_segs} segments)",
        fontsize=11,
    )
    fig.tight_layout()
    path = plot_dir / f"video_{base_col}_dist.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Metric distribution plot saved: %s", path)
    return path


def _compute_global_metric_stats(
    all_raw: list[pd.DataFrame],
    base_col: str,
) -> pd.DataFrame:
    """Global statistics (all frames, all segments) for columns matching ``base_col``.

    Per-keypoint columns (``_speed_kp_*``) and boolean flags are excluded.

    Parameters
    ----------
    all_raw : list[pd.DataFrame]
    base_col : str

    Returns
    -------
    pd.DataFrame
        Index = metric column names.
        Columns: N_valid, global_mean, global_std, global_median, pct_valid_%.
        Empty DataFrame if no matching columns.
    """
    sample_cols: list[str] = []
    for df in all_raw:
        for c in df.columns:
            if (
                base_col in c
                and c not in sample_cols
                and not c.endswith("_changed")
                and "_speed_kp_" not in c
            ):
                sample_cols.append(c)

    if not sample_cols:
        return pd.DataFrame()

    total_frames = sum(len(df) for df in all_raw)
    all_vals: dict[str, list] = {c: [] for c in sample_cols}
    for df in all_raw:
        for c in sample_cols:
            if c in df.columns:
                all_vals[c].extend(df[c].dropna().tolist())

    rows = []
    for c in sample_cols:
        v = np.array(all_vals[c], dtype=float)
        rows.append({
            "metric": c,
            "N_valid": len(v),
            "global_mean":   float(np.mean(v))   if len(v) > 0 else np.nan,
            "global_std":    float(np.std(v))    if len(v) > 0 else np.nan,
            "global_median": float(np.median(v)) if len(v) > 0 else np.nan,
            "pct_valid_%":   100.0 * len(v) / max(total_frames, 1),
        })
    return pd.DataFrame(rows).set_index("metric")


# ---- Individual plot helpers ----

def _plot_speed_panel(
    plot_dir: Path, raw: pd.DataFrame, norm: pd.DataFrame, level: str
) -> list[Path]:
    saved = []
    ind_names = _detect_individuals(raw, "speed_centroid")
    for variant in ("centroid", "trunk"):
        cols = [c for c in raw.columns if f"speed_{variant}" in c and not c.endswith("changed")]
        if not cols:
            continue
        fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TALL, sharex=True)
        for ax, df, label in zip(axes, [raw, norm], ["Raw (px/frame)", "Normalised (trunk-height/frame)"]):
            for col in cols:
                if col in df.columns:
                    ind = col.split("_speed")[0]
                    ax.plot(df.index, df[col], label=ind, color=_color(ind), linewidth=0.8, alpha=0.85)
            ax.set_ylabel(label)
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        axes[-1].set_xlabel("Frame")
        axes[0].set_title(f"{METRIC_META.get(f'speed_{variant}', {}).get('title', f'Speed ({variant})')} — {level}")
        fig.tight_layout()
        path = plot_dir / f"speed_{variant}.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def _plot_acceleration_panel(plot_dir: Path, raw: pd.DataFrame, level: str) -> list[Path]:
    saved = []
    for variant in ("centroid", "trunk"):
        cols = [c for c in raw.columns if f"acceleration_{variant}" in c]
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        for col in cols:
            ind = col.split("_acceleration")[0]
            ax.plot(raw.index, raw[col], label=ind, color=_color(ind), linewidth=0.8, alpha=0.85)
        ax.set_xlabel("Frame"); ax.set_ylabel("px/frame²")
        ax.legend(fontsize=8)
        ax.set_title(f"Acceleration ({variant}) — {level}")
        fig.tight_layout()
        path = plot_dir / f"acceleration_{variant}.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def _plot_ke_panel(
    plot_dir: Path, raw: pd.DataFrame, norm: pd.DataFrame, level: str
) -> list[Path]:
    ke_cols = [c for c in raw.columns if "kinetic_energy" in c]
    if not ke_cols:
        return []
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TALL, sharex=True)
    for ax, df, label in zip(axes, [raw, norm], ["Raw (px²)", "Normalised (trunk-height²)"]):
        for col in ke_cols:
            ind = col.split("_kinetic")[0]
            ax.plot(df.index, df[col], label=ind, color=_color(ind), linewidth=0.8, alpha=0.85)
        if "agitation_global_ke" in df.columns:
            ax.plot(df.index, df["agitation_global_ke"], label="global", color=_color("global"),
                    linewidth=1.2, linestyle="--")
        ax.set_ylabel(label); ax.legend(fontsize=8)
    axes[-1].set_xlabel("Frame")
    axes[0].set_title(f"Kinetic Energy / Agitation — {level}")
    fig.tight_layout()
    path = plot_dir / "kinetic_energy.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _plot_interpersonal(
    plot_dir: Path, raw: pd.DataFrame, norm: pd.DataFrame, level: str
) -> list[Path]:
    saved = []
    for col in ("interpersonal_distance_centroid", "interpersonal_distance_trunk",
                "interpersonal_approach"):
        if col not in raw.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
        meta = METRIC_META.get(col, {})
        for ax, df, label in zip(axes, [raw, norm], ["Raw", "Normalised"]):
            if col in df.columns:
                ax.plot(df.index, df[col], color="#795548", linewidth=0.8)
                ax.set_xlabel("Frame"); ax.set_ylabel(label)
                ax.set_title(f"{meta.get('title', col)} — {label}")
        fig.suptitle(f"{meta.get('title', col)} — {level}")
        fig.tight_layout()
        path = plot_dir / f"{col}.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def _plot_facingness(plot_dir: Path, raw: pd.DataFrame, level: str) -> list[Path]:
    if "facingness" not in raw.columns:
        return []
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(raw.index, raw["facingness"], color="#FF5722", linewidth=0.8, alpha=0.85)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axhline(1, color="grey", linewidth=0.5, linestyle="--", label="+1 (same direction)")
    ax.axhline(-1, color="grey", linewidth=0.5, linestyle="--", label="-1 (face-to-face)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Frame"); ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Facingness — {level}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = plot_dir / "facingness.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _plot_congruent(plot_dir: Path, raw: pd.DataFrame, level: str) -> list[Path]:
    if "congruent_motion" not in raw.columns:
        return []
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(raw.index, raw["congruent_motion"], color="#009688", linewidth=0.9)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Frame"); ax.set_ylabel("Pearson r")
    ax.set_title(f"Congruent Motion (Synchrony) — {level}")
    fig.tight_layout()
    path = plot_dir / "congruent_motion.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _plot_overlay(
    plot_dir: Path,
    all_raw: list[pd.DataFrame],
    all_norm: list[pd.DataFrame],
    merged: pd.DataFrame,
) -> list[Path]:
    """Overlay plot: light per-segment lines + bold smoothed overall trend."""
    saved = []
    cols_of_interest = [
        "speed_centroid", "speed_trunk",
        "kinetic_energy", "agitation_global_ke",
        "acceleration_centroid",
        "interpersonal_distance_centroid", "interpersonal_approach",
        "facingness", "congruent_motion",
    ]

    for base_col in cols_of_interest:
        # Find all columns containing this base name
        matching = [c for c in merged.columns if base_col in c and "segment_id" not in c]
        if not matching:
            continue

        fig, ax = plt.subplots(figsize=(16, 4))
        for col in matching:
            ind = col.split(f"_{base_col}")[0] if f"_{base_col}" in col else base_col
            color = _color(ind)
            # Per-segment (light)
            for i, seg_df in enumerate(all_raw):
                if col in seg_df.columns:
                    ax.plot(seg_df.index, seg_df[col], color=color, alpha=0.25, linewidth=0.5)
            # Overall smoothed trend
            valid_vals = merged[col].dropna()
            if len(valid_vals) > 30:
                smoothed = _smooth(merged[col].values, window=31)
                ax.plot(np.arange(len(smoothed)), smoothed, color=color, linewidth=1.5,
                        label=ind, alpha=0.9)

        ax.set_xlabel("Frame (continuous)"); ax.set_ylabel("")
        meta = METRIC_META.get(base_col, {})
        ax.set_title(f"{meta.get('title', base_col)} — video overview")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / f"video_{base_col}.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average for visualisation (ignores NaN)."""
    result = np.full_like(arr, np.nan, dtype=float)
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        vals = arr[lo:hi]
        valid = vals[~np.isnan(vals.astype(float))]
        if len(valid) > 0:
            result[i] = float(np.mean(valid))
    return result


def _detect_individuals(df: pd.DataFrame, metric: str) -> list[str]:
    return list({c.split(f"_{metric}")[0] for c in df.columns if f"_{metric}" in c})


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def generate_html_report(
    video_name: str,
    video_dir: Path,
    stats: CleaningStats,
    all_raw: list[pd.DataFrame],
    fps: float,
) -> Path:
    """Generate a self-contained HTML report for a processed video.

    Parameters
    ----------
    video_name : str
    video_dir : Path
    stats : CleaningStats
    all_raw : list of per-segment raw DataFrames
    fps : float

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    logger.info("Generating HTML report for '%s' …", video_name)

    plot_dir = video_dir / "plots"
    sections: list[str] = []

    # ---- Overview ----
    total_valid = sum(len(df) for df in all_raw)
    total_dur_s = stats.total_frames / max(fps, 1)
    valid_dur_s = total_valid / max(fps, 1)
    sections.append(_section_overview(
        video_name, stats, fps, total_dur_s, valid_dur_s,
    ))

    # ---- Cleaning report ----
    sections.append(_section_cleaning(stats, fps))

    # ---- Segment inventory (with optional length-distribution plot) ----
    seg_dist_plot = plot_dir / "segment_length_dist.png"
    sections.append(_section_segments(
        stats, fps,
        seg_dist_plot if seg_dist_plot.exists() else None,
    ))

    # ---- Per-metric sections ----
    for base_col, meta in METRIC_META.items():
        # Overlay time-series plot (one per base metric)
        overlay_path = plot_dir / f"video_{base_col}.png"
        overlay_plots = [overlay_path] if overlay_path.exists() else []
        # Distribution plot (per-segment means & medians)
        dist_path = plot_dir / f"video_{base_col}_dist.png"
        dist_plot = dist_path if dist_path.exists() else None
        # Global statistics across all frames
        global_stats = _compute_global_metric_stats(all_raw, base_col)
        sections.append(_section_metric(meta, overlay_plots, dist_plot, global_stats, base_col))

    # ---- Keypoint comparison section ----
    kp_comp_plots = sorted(plot_dir.glob("video_kp_comparison_*.png")) if plot_dir.exists() else []
    sections.append(_section_kp_comparison_html(kp_comp_plots))

    # ---- Assemble document ----
    html = _html_wrapper(video_name, "\n".join(sections))
    out_path = video_dir / f"report_{video_name}.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", out_path)
    return out_path


# ---- HTML building blocks ----

def _section_overview(
    video_name: str, stats: CleaningStats, fps: float,
    total_dur_s: float, valid_dur_s: float,
) -> str:
    segs_info = ", ".join(f"{s} frames ({s/fps:.1f}s)" for s in stats.segment_lengths)
    return f"""
<section id="overview">
  <h2>Overview: {_esc(video_name)}</h2>
  <table class="summary-table">
    <tr><th>Total frames</th><td>{stats.total_frames}</td></tr>
    <tr><th>Total duration</th><td>{total_dur_s:.1f} s ({total_dur_s/60:.1f} min)</td></tr>
    <tr><th>FPS</th><td>{fps}</td></tr>
    <tr><th>Dyadic individuals</th><td>{", ".join(stats.dyadic_individuals)}</td></tr>
    <tr><th>Confidence threshold</th><td>{stats.conf_threshold}</td></tr>
    <tr><th>Min valid keypoints</th><td>{stats.min_valid_kp}</td></tr>
    <tr><th>Min segment frames</th><td>{stats.min_segment_frames} ({stats.min_segment_frames/fps:.1f}s)</td></tr>
    <tr><th>Segments found</th><td>{stats.segments_found}</td></tr>
    <tr><th>Segments kept</th><td>{stats.segments_kept}</td></tr>
    <tr><th>Total valid frames</th><td>{sum(stats.segment_lengths)} ({valid_dur_s:.1f}s)</td></tr>
    <tr><th>Segment lengths</th><td>{segs_info or "—"}</td></tr>
  </table>
</section>
"""


def _section_cleaning(stats: CleaningStats, fps: float) -> str:
    rows = [
        ("original", stats.total_frames, 0, 0.0),
        ("after confidence filter", stats.frames_after_conf,
         stats.total_frames - stats.frames_after_conf,
         100.0 * (stats.total_frames - stats.frames_after_conf) / max(stats.total_frames, 1)),
        ("after dyadic filter", stats.frames_after_dyadic,
         stats.total_frames - stats.frames_after_dyadic,
         100.0 * (stats.total_frames - stats.frames_after_dyadic) / max(stats.total_frames, 1)),
        ("after continuity / min-length filter", stats.frames_after_continuity,
         stats.total_frames - stats.frames_after_continuity,
         100.0 * (stats.total_frames - stats.frames_after_continuity) / max(stats.total_frames, 1)),
    ]
    table_rows = "\n".join(
        f"<tr><td>{step}</td><td>{frames}</td><td>{dropped}</td><td>{pct:.1f}%</td></tr>"
        for step, frames, dropped, pct in rows
    )

    # Coverage table — colour-code the delta:
    #   green  : small positive Δ (expected — sample-selection bias, see note)
    #   grey   : near zero (< 1 pp either way)
    #   orange : moderate decrease −1 % to −20 %  (confidence filtering, expected)
    #   red    : large decrease > −20 % (may warrant inspection)
    cov_rows = ""
    for ind in stats.kp_coverage_before:
        for kp in stats.kp_coverage_before[ind]:
            b = stats.kp_coverage_before[ind][kp]
            a = stats.kp_coverage_after.get(ind, {}).get(kp, 0.0)
            delta = a - b
            if abs(delta) < 1.0:
                color = "#888"          # near-zero — grey
            elif delta > 0:
                color = "#2E7D32"       # positive — dark green (expected)
            elif delta > -20.0:
                color = "#E65100"       # moderate drop — orange (expected)
            else:
                color = "#B71C1C"       # large drop — red (worth checking)
            cov_rows += (
                f"<tr><td>{_esc(ind)}</td><td>{_esc(kp)}</td>"
                f"<td>{b:.1f}%</td><td>{a:.1f}%</td>"
                f"<td style='color:{color}; font-weight:bold'>{delta:+.1f}%</td></tr>\n"
            )

    return f"""
<section id="cleaning">
  <h2>Cleaning Report</h2>
  <p>Numbers show cumulative frames <em>kept</em> after each step.</p>
  <table class="summary-table">
    <tr><th>Step</th><th>Frames kept</th><th>Frames dropped (cumul.)</th><th>% dropped</th></tr>
    {table_rows}
  </table>

  <h3>Keypoint Coverage: Before vs After Cleaning</h3>

  <div class="metric-box" style="background:#FFF8E1; border-color:#F9A825;">
    <p><strong>How coverage is defined</strong></p>
    <p>
      <em>Coverage</em> = number of frames where the keypoint is <strong>visible</strong>
      (position ≠ [0, 0]) ÷ number of frames where the <strong>individual is present</strong>
      (at least one non-zero keypoint).  Frames where the individual is entirely absent
      are excluded from both numerator and denominator, so individual absence does not
      artificially deflate keypoint coverage.
    </p>
    <p><strong>Before</strong> — computed over every frame in the full raw recording
      where the individual appears.</p>
    <p><strong>After</strong> — computed over only the <em>kept segments</em>: the
      contiguous blocks that survived all three cleaning stages.</p>

    <p><strong>Why Δ can be positive (coverage increases after cleaning)</strong></p>
    <p>
      The kept segments are not a random sample of the full recording — they are the
      periods where <em>both</em> dyadic individuals were reliably detected with
      sufficient keypoints.  These are inherently higher-quality frames.  Rare or
      unstable keypoints (e.g. <em>left_ankle</em>) that only appear sporadically in
      the full recording will naturally seem more consistently visible within those
      selected high-quality blocks.  A positive Δ is therefore <strong>expected and
      desirable</strong>: it confirms the cleaning isolated the well-detected periods.
    </p>

    <p><strong>Why Δ can be negative (coverage decreases after cleaning)</strong></p>
    <p>
      The confidence filter (stage 2) zeros out keypoints whose detection score falls
      below the threshold ({stats.conf_threshold}).  Keypoints that were visible in the
      raw tracking but with low confidence will disappear after this filter, reducing
      their coverage even within the kept segments.  A moderate negative Δ (up to
      ≈ −20 pp) is normal and reflects appropriate removal of unreliable detections.
      A very large negative Δ on a core keypoint (e.g. shoulder, hip) may indicate
      that the confidence threshold is too aggressive for that individual or camera view.
    </p>

    <p style="font-size:0.85em; color:#555;">
      Colour guide:
      <span style="color:#2E7D32; font-weight:bold">Green = positive Δ</span> (expected, sample-selection bias) &nbsp;|&nbsp;
      <span style="color:#888; font-weight:bold">Grey ≈ 0</span> &nbsp;|&nbsp;
      <span style="color:#E65100; font-weight:bold">Orange = moderate drop</span> (confidence filter, expected) &nbsp;|&nbsp;
      <span style="color:#B71C1C; font-weight:bold">Red = large drop &gt; 20 pp</span> (worth inspecting)
    </p>
  </div>

  <table class="summary-table">
    <tr><th>Individual</th><th>Keypoint</th><th>Before</th><th>After</th><th>Δ</th></tr>
    {cov_rows}
  </table>
</section>
"""


def _section_metric(
    meta: dict,
    overlay_plots: list[Path],
    dist_plot: "Path | None",
    global_stats: pd.DataFrame,
    base_col: str,
) -> str:
    """HTML section for a single metric.

    Content order:

    1. Definition / interpretation box.
    2. Distribution plot — per-segment means and medians (one dot per segment).
    3. Overlay time-series — all segments (light) + smoothed overall trend (bold).
    4. Compact global statistics table (N valid frames, mean, std, median).
    """
    title          = meta.get("title", base_col)
    definition     = meta.get("definition", "")
    interpretation = meta.get("interpretation", "")
    unit           = meta.get("unit", "")

    # 1. Distribution plot (per-segment means / medians)
    dist_img = ""
    if dist_plot is not None:
        b64 = _png_to_base64(dist_plot)
        dist_img = (
            "<h4>Distribution of per-segment means and medians</h4>\n"
            "<p style='font-size:0.85em;color:#555;'>"
            "Each dot = one segment.  Box = IQR.  "
            "Dashed line = global value across all frames.</p>\n"
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="distribution of {_esc(base_col)}" style="max-width:100%;"/>\n'
        )

    # 2. Overlay time-series
    overlay_imgs = ""
    for p in overlay_plots:
        b64 = _png_to_base64(p)
        overlay_imgs += (
            "<h4>Time-series overview (all segments, smoothed trend)</h4>\n"
            f'<img src="data:image/png;base64,{b64}" alt="{_esc(p.name)}" '
            f'style="max-width:100%;"/>\n'
        )

    # 3. Compact global statistics table
    global_stats_html = ""
    if not global_stats.empty:
        global_stats_html = (
            "<h4>Global statistics — all frames, all segments</h4>\n"
            + global_stats.round(4).to_html(classes="summary-table", border=0)
        )

    return f"""
<section id="metric-{base_col.replace('_', '-')}">
  <h2 title="{_esc(interpretation)}">{_esc(title)} ℹ</h2>
  <div class="metric-box">
    <p><strong>Definition:</strong> {_esc(definition)}</p>
    <p><strong>Unit:</strong> {_esc(unit)}</p>
    <p class="tooltip" title="{_esc(interpretation)}">
      <strong>How to interpret:</strong> hover here for interpretation guide.
    </p>
  </div>
  {dist_img}
  {overlay_imgs}
  {global_stats_html}
</section>
"""


def _section_segments(
    stats: CleaningStats,
    fps: float,
    seg_dist_plot: "Path | None" = None,
) -> str:
    """HTML section listing kept segments with start/end/duration info.

    Parameters
    ----------
    stats : CleaningStats
    fps : float
    seg_dist_plot : Path or None
        Optional path to the segment-length distribution PNG to embed below
        the inventory table.
    """
    rows_html = ""
    for i, (start, end, length) in enumerate(
        zip(stats.segment_starts, stats.segment_ends, stats.segment_lengths), start=1
    ):
        dur_s = length / max(fps, 1)
        rows_html += (
            f"<tr><td>seg_{i:03d}</td>"
            f"<td>{start}</td><td>{end}</td>"
            f"<td>{length}</td><td>{dur_s:.1f} s</td></tr>\n"
        )

    total_frames = sum(stats.segment_lengths)
    total_dur_s = total_frames / max(fps, 1)
    rows_html += (
        f"<tr style='font-weight:bold; background:#BBDEFB'>"
        f"<td>TOTAL</td><td>—</td><td>—</td>"
        f"<td>{total_frames}</td><td>{total_dur_s:.1f} s</td></tr>\n"
    )

    dist_img = ""
    if seg_dist_plot is not None:
        b64 = _png_to_base64(seg_dist_plot)
        dist_img = (
            "<h3>Segment Length Distribution</h3>\n"
            "<p style='font-size:0.85em;color:#555;'>"
            "Histogram of segment durations.  Rug marks at the bottom show "
            "each individual segment.  Dashed red = mean, dotted orange = "
            "median, dash-dot grey = minimum-length threshold.</p>\n"
            '<img src="data:image/png;base64,' + b64 + '" '
            'alt="segment length distribution" '
            'style="max-width:90%; display:block; margin:10px auto;"/>\n'
        )

    return f"""
<section id="segments">
  <h2>Segment Inventory</h2>
  <p>
    Each segment is a contiguous block of valid frames (both dyadic individuals
    present with sufficient keypoint confidence).  Short segments below the
    minimum length threshold were discarded before this table.
    Segments are re-indexed from frame 0 when saved as individual
    <code>tracking.nc</code> files; the <em>Start / End</em> columns show the
    original frame numbers in the raw video.
  </p>
  <table class="summary-table">
    <tr>
      <th>Segment ID</th>
      <th>Start frame (orig.)</th>
      <th>End frame (orig.)</th>
      <th>Length (frames)</th>
      <th>Duration</th>
    </tr>
    {rows_html}
  </table>
  <p style="font-size:0.85em; color:#666;">
    Segments found before length filter: {stats.segments_found} &nbsp;|&nbsp;
    Kept: {stats.segments_kept} &nbsp;|&nbsp;
    Min segment length: {stats.min_segment_frames} frames
    ({stats.min_segment_frames / max(fps, 1):.1f} s)
  </p>
  {dist_img}
</section>
"""


def _plot_kp_comparison(
    plot_dir: Path,
    all_raw: list[pd.DataFrame],
    merged: pd.DataFrame,
) -> list[Path]:
    """Plot centroid speed − trunk speed difference per dyadic individual.

    A single figure with one subplot per individual shows the smoothed
    difference signal ``centroid_speed − trunk_speed`` over time.

    * **Positive** → the whole-body centroid moves faster than the trunk alone.
      This can indicate genuine limb motion (e.g. arm swing) or noise from
      unreliable extremity keypoints inflating the centroid displacement.
    * **Negative** → trunk moves faster than the whole-body centroid (rare;
      suggests that extremity keypoints are dragging the centroid toward a
      stable position even though the trunk is moving).
    * **Near zero** → centroid and trunk agree — the movement signal is
      consistent across the body.

    Vertical dashed lines mark segment boundaries.
    A dashed horizontal line shows the global mean difference.

    Returns
    -------
    list[Path]
        Single-element list containing the saved PNG path, or empty list if
        no centroid/trunk speed columns are found.
    """
    # Detect individuals with both centroid and trunk speed columns
    inds = [
        c.split("_speed_centroid")[0]
        for c in merged.columns
        if "_speed_centroid" in c
        and f"{c.split('_speed_centroid')[0]}_speed_trunk" in merged.columns
    ]
    inds = list(dict.fromkeys(inds))  # preserve order, deduplicate

    if not inds:
        return []

    n_inds = len(inds)
    fig, axes = plt.subplots(n_inds, 1, figsize=(16, 4 * n_inds), sharex=True,
                              squeeze=False)

    for ax, ind in zip(axes[:, 0], inds):
        centroid_col = f"{ind}_speed_centroid"
        trunk_col    = f"{ind}_speed_trunk"
        color        = _color(ind)

        centroid_vals = merged[centroid_col].values.astype(float)
        trunk_vals    = merged[trunk_col].values.astype(float)
        diff_vals     = centroid_vals - trunk_vals
        diff_s        = _smooth(diff_vals, window=31)

        x = np.arange(len(merged))

        # Shaded fill: positive (centroid > trunk) vs negative (trunk > centroid)
        ax.fill_between(x, diff_s, 0.0,
                        where=(diff_s >= 0), alpha=0.20, color=color,
                        label="Centroid > trunk (extremities add speed)")
        ax.fill_between(x, diff_s, 0.0,
                        where=(diff_s < 0),  alpha=0.20, color="#F44336",
                        label="Trunk > centroid (extremities dampen centroid)")

        # Difference signal
        ax.plot(x, diff_s, color=color, linewidth=1.5, alpha=0.9)

        # Zero reference
        ax.axhline(0.0, color="#555", linewidth=0.8, linestyle=":")

        # Global mean difference
        global_mean = float(np.nanmean(diff_vals))
        ax.axhline(global_mean, color=color, linewidth=1.0, linestyle="--", alpha=0.7,
                   label=f"Global mean diff: {global_mean:+.4f} px/frame")

        # Segment boundary markers (first frame index of each segment)
        seg_starts = (
            merged.drop_duplicates(subset="segment_id", keep="first").index.values
        )
        for b in seg_starts[1:]:
            ax.axvline(b, color="#bbb", linewidth=0.5, linestyle="--")

        ax.set_ylabel("Speed diff (px/frame)", fontsize=9)
        ax.set_title(
            f"Centroid Speed − Trunk Speed — {ind}",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)

    axes[-1, 0].set_xlabel("Frame (continuous across segments)", fontsize=10)
    fig.suptitle(
        "Keypoint Speed Comparison: Centroid − Trunk Speed\n"
        "Positive = extremities inflate centroid  |  Zero = perfect agreement  |  "
        "Smoothed (31-frame window)",
        fontsize=11,
    )
    fig.tight_layout()

    path = plot_dir / "video_kp_comparison_diff.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.debug("KP difference comparison plot saved: %s", path)
    return [path]


def _section_kp_comparison_html(plots: list[Path]) -> str:
    """HTML section for the centroid − trunk speed difference plot."""
    img_tags = "\n".join(
        f'<img src="data:image/png;base64,{_png_to_base64(p)}" '
        f'alt="{p.name}" style="max-width:100%;"/>'
        for p in plots
    )
    return f"""
<section id="kp-comparison">
  <h2>Keypoint Speed Comparison: Centroid − Trunk Speed ℹ</h2>
  <div class="metric-box">
    <p><strong>What is shown:</strong>
      For each dyadic individual, the plot shows the smoothed difference
      <code>centroid_speed − trunk_speed</code> over time.
    </p>
    <p><strong>Centroid speed</strong> uses the intersection of <em>all</em>
      visible keypoints at frames t and t−1 to compute the displacement of the
      mean body position.
      <strong>Trunk speed</strong> restricts the same computation to the four
      trunk keypoints (left/right shoulder and left/right hip), which are
      typically the most stably detected.
    </p>
    <p><strong>How to interpret the difference signal:</strong></p>
    <ul>
      <li><strong>Near zero (flat baseline)</strong> — centroid and trunk
        speeds agree.  The movement signal is consistent across the body;
        the centroid metric is <em>reliable</em>.</li>
      <li><strong>Persistently positive</strong> — extremity keypoints
        (wrists, ankles, elbows, knees) are moving faster than the trunk, or
        noisy extremity detections are inflating the centroid displacement.
        Consider using trunk speed as the primary metric for this individual.</li>
      <li><strong>Persistently negative</strong> (rare) — the trunk is moving
        faster than the whole-body centroid, suggesting extremity positions
        are partially anchoring the centroid.</li>
      <li><strong>Transient spikes</strong> — often coincide with a
        <code>kp_set_changed</code> event (a keypoint appearing or disappearing
        causes a centroid jump unrelated to actual movement).</li>
    </ul>
    <p title="Difference smoothed with a 31-frame rolling mean for readability.
Values are in raw pixel units (not trunk-height normalised).
Vertical dashed lines mark segment boundaries.
Dashed horizontal line = global mean difference.">
      Smoothed (31-frame window).  Hover for technical notes.
    </p>
  </div>
  {img_tags if img_tags else "<p><em>No comparison plots available.</em></p>"}
</section>
"""


def _html_wrapper(video_name: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Quality Report — {_esc(video_name)}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px;
          background: #fafafa; color: #333; }}
  h1 {{ color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }}
  h2 {{ color: #0D47A1; margin-top: 40px; }}
  h3 {{ color: #1976D2; }}
  section {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
             box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  .summary-table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  .summary-table th {{ background: #E3F2FD; }}
  .summary-table tr:nth-child(even) {{ background: #F5F5F5; }}
  .metric-box {{ background: #E8F5E9; border-left: 4px solid #388E3C; padding: 12px;
                 border-radius: 4px; margin-bottom: 12px; }}
  .tooltip {{ cursor: help; border-bottom: 1px dashed #888; display: inline-block; }}
  img {{ display: block; margin: 10px auto; border-radius: 4px;
         box-shadow: 0 1px 4px rgba(0,0,0,0.12); }}
  nav {{ background: #1565C0; color: white; padding: 10px 20px; border-radius: 6px;
         margin-bottom: 20px; }}
  nav a {{ color: #BBDEFB; margin-right: 16px; text-decoration: none; font-size: 0.9em; }}
  nav a:hover {{ color: white; text-decoration: underline; }}
</style>
</head>
<body>
<h1>Pose Quality Report — {_esc(video_name)}</h1>
<nav>
  <a href="#overview">Overview</a>
  <a href="#cleaning">Cleaning</a>
  <a href="#segments">Segments</a>
  {''.join(f'<a href="#metric-{k.replace("_","-")}">{v["title"]}</a>' for k, v in METRIC_META.items())}
  <a href="#kp-comparison">KP Comparison</a>
</nav>
{body}
<footer style="margin-top:40px; color:#999; font-size:0.8em; text-align:center;">
  Generated by poseToRecord — {video_name}
</footer>
</body>
</html>"""


def _png_to_base64(path: Path) -> str:
    with path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
