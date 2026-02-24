"""Filtering and cleaning pipeline for pose-tracking datasets.

This module implements the full cleaning pipeline that transforms a raw
:class:`xarray.Dataset` (as produced by :mod:`convert`) into cleaned records
ready for downstream analysis.

Pipeline stages (applied in order)
------------------------------------
1. **Negative-coordinate cropping** – clip ``position`` values to ``[0, +∞)``.
2. **Confidence filtering** – zero out keypoints whose confidence score is
   below ``conf_threshold``.
3. **Dyadic-presence filter** – mark frames invalid when one or both
   *dyadic individuals* do not have enough high-confidence keypoints.
4. **Continuity detection** – find runs of consecutive valid frames and
   discard segments shorter than ``min_segment_frames``.
5. **Auto-scaling** – normalise valid positions to the ``[0, 1]`` range
   (zero values that encode missing keypoints remain zero).

In addition to the cleaned dataset(s), the pipeline returns detailed
**cleaning statistics** that document how many frames were removed at each
step and how keypoint coverage changed.

Convention
----------
A position of ``[0.0, 0.0]`` encodes a *missing* keypoint throughout the
entire ``poseToRecord`` codebase.  This convention matches the one used in
``bunch_of_stuff.py`` / ``_load_posetracks``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class Segment(NamedTuple):
    """A contiguous range of valid frames.

    Attributes
    ----------
    start : int
        First valid frame index (inclusive).
    end : int
        Last valid frame index (inclusive).
    """

    start: int
    end: int

    @property
    def length(self) -> int:
        """Number of frames in the segment."""
        return self.end - self.start + 1


@dataclass
class CleaningStats:
    """Detailed statistics produced by :func:`apply_cleaning_pipeline`.

    Attributes
    ----------
    total_frames : int
        Total frames in the raw dataset.
    frames_after_conf : int
        Frames remaining after confidence filtering (frames where at least one
        individual has at least one visible keypoint).
    frames_after_dyadic : int
        Valid frames after dyadic-presence filtering.
    frames_after_continuity : int
        Frames inside continuous segments of sufficient length.
    segments_found : int
        Number of continuous segments detected before length filtering.
    segments_kept : int
        Number of segments kept after length filtering.
    segment_lengths : list[int]
        Length (in frames) of each *kept* segment.
    kp_coverage_before : dict[str, dict[str, float]]
        ``{individual_name: {keypoint_name: coverage_pct}}`` **before**
        cleaning (raw confidence-based coverage).
    kp_coverage_after : dict[str, dict[str, float]]
        Same structure **after** the full cleaning pipeline.
    conf_threshold : float
    min_valid_kp : int
    min_segment_frames : int
    dyadic_individuals : list[str]
    """

    total_frames: int = 0
    frames_after_conf: int = 0
    frames_after_dyadic: int = 0
    frames_after_continuity: int = 0
    segments_found: int = 0
    segments_kept: int = 0
    segment_lengths: list[int] = field(default_factory=list)
    segment_starts: list[int] = field(default_factory=list)
    segment_ends: list[int] = field(default_factory=list)
    kp_coverage_before: dict[str, dict[str, float]] = field(default_factory=dict)
    kp_coverage_after: dict[str, dict[str, float]] = field(default_factory=dict)
    conf_threshold: float = 0.3
    min_valid_kp: int = 5
    min_segment_frames: int = 300
    dyadic_individuals: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_cleaning_pipeline(
    ds: xr.Dataset,
    dyadic_individuals: list[str],
    conf_threshold: float = 0.3,
    min_valid_kp: int = 5,
    min_segment_frames: int = 300,
) -> tuple[xr.Dataset, list[xr.Dataset], CleaningStats]:
    """Run the full cleaning pipeline on a raw pose-tracking dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Raw dataset with ``position`` (T, I, K, 2) and ``confidence``
        (T, I, K) variables, as returned by :func:`convert.convert_json_to_dataset`.
    dyadic_individuals : list[str]
        Names of the two individuals that must **both** be present for a
        frame to be considered valid.  Names must match entries in
        ``ds.coords["individuals"]``.
    conf_threshold : float, optional
        Confidence score below which a keypoint is zeroed out.  Default 0.3.
    min_valid_kp : int, optional
        Minimum number of keypoints (above threshold) required per dyadic
        individual for a frame to pass the dyadic filter.  Default 5.
    min_segment_frames : int, optional
        Segments shorter than this are discarded.  Default 300 (≈15 s at
        20 fps).

    Returns
    -------
    full_ds : xarray.Dataset
        The complete filtered dataset (all frames, invalid frames zeroed out).
        Time-indexed from 0 to T−1.  Suitable for saving as the "full record".
    segment_datasets : list[xarray.Dataset]
        One dataset per kept continuous segment.  Each is a *slice* of
        ``full_ds`` with ``time`` re-indexed starting at 0 (time-translated).
    stats : CleaningStats
        Detailed statistics describing what was removed at each step.

    Notes
    -----
    * The returned datasets still contain the ``confidence`` variable so that
      downstream code can re-apply custom thresholds if needed.
    * Auto-scaling (to [0, 1]) is applied to the segment datasets but **not**
      to the full record (which retains raw pixel coordinates after clipping).
    """
    available = ds.coords["individuals"].values.tolist()
    for ind in dyadic_individuals:
        if ind not in available:
            raise ValueError(
                f"Dyadic individual '{ind}' not found in dataset individuals: "
                f"{available}"
            )

    stats = CleaningStats(
        conf_threshold=conf_threshold,
        min_valid_kp=min_valid_kp,
        min_segment_frames=min_segment_frames,
        dyadic_individuals=dyadic_individuals,
    )
    stats.total_frames = int(ds.sizes["time"])
    logger.info("=== Cleaning pipeline start: %d frames ===", stats.total_frames)

    # ------------------------------------------------------------------
    # Stage 0: record keypoint coverage BEFORE cleaning
    # ------------------------------------------------------------------
    stats.kp_coverage_before = _compute_kp_coverage(ds)

    # ------------------------------------------------------------------
    # Stage 1: Clip negative coordinates
    # ------------------------------------------------------------------
    ds = _clip_negative_coords(ds)

    # ------------------------------------------------------------------
    # Stage 2: Confidence filtering
    # ------------------------------------------------------------------
    ds = _apply_confidence_filter(ds, conf_threshold)

    # Count frames where at least one individual has ≥1 visible keypoint
    pos = ds["position"].values  # (T, I, K, 2)
    visible_per_frame = (np.abs(pos).sum(axis=(1, 2, 3)) > 0)  # (T,)
    stats.frames_after_conf = int(visible_per_frame.sum())
    logger.info(
        "After confidence filter (threshold=%.2f): %d / %d frames with data",
        conf_threshold, stats.frames_after_conf, stats.total_frames,
    )

    # ------------------------------------------------------------------
    # Stage 3: Dyadic-presence filter
    # ------------------------------------------------------------------
    valid_mask = _compute_dyadic_mask(ds, dyadic_individuals, conf_threshold, min_valid_kp)
    stats.frames_after_dyadic = int(valid_mask.sum())
    dropped_dyadic = stats.total_frames - stats.frames_after_dyadic
    logger.info(
        "After dyadic filter (%s, min_valid_kp=%d): %d / %d valid frames "
        "(%d dropped, %.1f%%)",
        dyadic_individuals, min_valid_kp,
        stats.frames_after_dyadic, stats.total_frames,
        dropped_dyadic, 100.0 * dropped_dyadic / max(stats.total_frames, 1),
    )

    # Zero out invalid frames in the full dataset
    full_ds = _zero_invalid_frames(ds, valid_mask)

    # ------------------------------------------------------------------
    # Stage 4: Continuity detection & segment filtering
    # ------------------------------------------------------------------
    all_segments = find_continuous_segments(valid_mask)
    stats.segments_found = len(all_segments)
    logger.info("Continuous segments found: %d", stats.segments_found)

    kept_segments: list[Segment] = []
    for seg in all_segments:
        if seg.length >= min_segment_frames:
            kept_segments.append(seg)
            logger.debug(
                "  Keeping segment [%d, %d] → %d frames", seg.start, seg.end, seg.length,
            )
        else:
            logger.debug(
                "  Dropping segment [%d, %d] → %d frames (< %d)",
                seg.start, seg.end, seg.length, min_segment_frames,
            )

    stats.segments_kept = len(kept_segments)
    stats.segment_lengths = [s.length for s in kept_segments]
    stats.segment_starts = [s.start for s in kept_segments]
    stats.segment_ends = [s.end for s in kept_segments]
    stats.frames_after_continuity = sum(stats.segment_lengths)
    logger.info(
        "Segments kept: %d / %d — total valid frames: %d",
        stats.segments_kept, stats.segments_found, stats.frames_after_continuity,
    )
    if kept_segments:
        logger.info(
            "Segment durations (frames): %s",
            [s.length for s in kept_segments],
        )

    # ------------------------------------------------------------------
    # Stage 5: Build per-segment datasets (auto-scaled to [0, 1])
    # ------------------------------------------------------------------
    segment_datasets: list[xr.Dataset] = []
    for seg in kept_segments:
        seg_ds = _extract_segment(full_ds, seg)
        seg_ds = _autoscale(seg_ds)
        segment_datasets.append(seg_ds)

    # ------------------------------------------------------------------
    # Stage 6: Record keypoint coverage AFTER cleaning
    # ------------------------------------------------------------------
    # Build a merged dataset of all kept segments to compute overall coverage
    if segment_datasets:
        merged_pos = np.concatenate(
            [s["position"].values for s in segment_datasets], axis=0
        )
        merged_conf = np.concatenate(
            [s["confidence"].values for s in segment_datasets], axis=0
        )
        dummy_ds = xr.Dataset(
            {
                "position": (("time", "individuals", "keypoints", "space"), merged_pos),
                "confidence": (("time", "individuals", "keypoints"), merged_conf),
            },
            coords={
                "time": np.arange(merged_pos.shape[0]),
                "individuals": full_ds.coords["individuals"].values,
                "keypoints": full_ds.coords["keypoints"].values,
                "space": ["x", "y"],
            },
        )
        stats.kp_coverage_after = _compute_kp_coverage(dummy_ds)
    else:
        stats.kp_coverage_after = {
            ind: {kp: 0.0 for kp in ds.coords["keypoints"].values.tolist()}
            for ind in ds.coords["individuals"].values.tolist()
        }
        logger.warning("No segments kept — output will be empty.")

    logger.info("=== Cleaning pipeline complete ===")
    _log_coverage_delta(stats)

    return full_ds, segment_datasets, stats


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _clip_negative_coords(ds: xr.Dataset) -> xr.Dataset:
    """Clip position coordinates to [0, +inf); log how many values changed."""
    pos = ds["position"].values.copy()
    n_negative = int((pos < 0).sum())
    if n_negative > 0:
        logger.info(
            "Clipping %d negative coordinate value(s) to 0.0", n_negative,
        )
        pos = np.clip(pos, 0.0, None)
    else:
        logger.debug("No negative coordinate values found.")
    return ds.assign(position=(ds["position"].dims, pos))


def _apply_confidence_filter(ds: xr.Dataset, conf_threshold: float) -> xr.Dataset:
    """Zero out keypoints whose confidence is below ``conf_threshold``.

    A zeroed keypoint has both its x and y set to 0.0 (missing convention).
    """
    conf = ds["confidence"].values  # (T, I, K)
    pos = ds["position"].values.copy()  # (T, I, K, 2)

    low_conf_mask = conf < conf_threshold  # (T, I, K)  True = below threshold
    pos[low_conf_mask] = 0.0             # broadcast over space dim

    n_zeroed = int(low_conf_mask.sum())
    logger.info(
        "Confidence filter: zeroed %d keypoint instance(s) (threshold=%.2f)",
        n_zeroed, conf_threshold,
    )

    # Per-individual, per-keypoint breakdown
    ind_names = ds.coords["individuals"].values.tolist()
    kp_names = ds.coords["keypoints"].values.tolist()
    n_frames = pos.shape[0]
    for i, ind in enumerate(ind_names):
        for k, kp in enumerate(kp_names):
            n_dropped = int(low_conf_mask[:, i, k].sum())
            if n_dropped > 0:
                logger.debug(
                    "  %s / %s: %d / %d frames zeroed (%.1f%%)",
                    ind, kp, n_dropped, n_frames,
                    100.0 * n_dropped / max(n_frames, 1),
                )

    return ds.assign(position=(ds["position"].dims, pos))


def _compute_dyadic_mask(
    ds: xr.Dataset,
    dyadic_individuals: list[str],
    conf_threshold: float,
    min_valid_kp: int,
) -> np.ndarray:
    """Return a boolean mask of shape (T,) — True = dyadic condition met.

    A frame is valid when **all** dyadic individuals have at least
    ``min_valid_kp`` keypoints with confidence >= ``conf_threshold``.
    """
    conf = ds["confidence"].values  # (T, I, K)
    ind_names = ds.coords["individuals"].values.tolist()
    valid_mask = np.ones(conf.shape[0], dtype=bool)

    for ind in dyadic_individuals:
        if ind not in ind_names:
            raise ValueError(f"Individual '{ind}' not in dataset.")
        idx = ind_names.index(ind)
        # Number of keypoints above threshold per frame
        n_valid = (conf[:, idx, :] >= conf_threshold).sum(axis=1)  # (T,)
        ind_mask = n_valid >= min_valid_kp
        n_invalid = int((~ind_mask).sum())
        logger.info(
            "  Dyadic check '%s': %d frames fail (< %d keypoints above %.2f)",
            ind, n_invalid, min_valid_kp, conf_threshold,
        )
        valid_mask &= ind_mask

    return valid_mask


def _zero_invalid_frames(ds: xr.Dataset, valid_mask: np.ndarray) -> xr.Dataset:
    """Zero out position (and confidence) for frames where ``valid_mask`` is False."""
    pos = ds["position"].values.copy()
    conf = ds["confidence"].values.copy()
    invalid = ~valid_mask
    pos[invalid] = 0.0
    conf[invalid] = 0.0
    return ds.assign(
        position=(ds["position"].dims, pos),
        confidence=(ds["confidence"].dims, conf),
    )


def find_continuous_segments(valid_mask: np.ndarray) -> list[Segment]:
    """Identify contiguous runs of ``True`` in a boolean mask.

    Parameters
    ----------
    valid_mask : np.ndarray, dtype bool, shape (T,)
        Per-frame validity flag.

    Returns
    -------
    list[Segment]
        Sorted list of :class:`Segment` objects (start, end inclusive).
    """
    segments: list[Segment] = []
    in_seg = False
    start = 0
    for t, v in enumerate(valid_mask):
        if v and not in_seg:
            start = t
            in_seg = True
        elif not v and in_seg:
            segments.append(Segment(start, t - 1))
            in_seg = False
    if in_seg:
        segments.append(Segment(start, len(valid_mask) - 1))
    return segments


def _extract_segment(ds: xr.Dataset, seg: Segment) -> xr.Dataset:
    """Slice a segment from the dataset and re-index time from 0."""
    seg_ds = ds.isel(time=slice(seg.start, seg.end + 1))
    # Re-index time to start from 0
    seg_ds = seg_ds.assign_coords(time=np.arange(seg.length, dtype=np.int64))
    seg_ds.attrs = {**ds.attrs, "segment_start_frame": seg.start, "segment_end_frame": seg.end}
    return seg_ds


def _autoscale(ds: xr.Dataset) -> xr.Dataset:
    """Normalise valid positions to [0, 1]; missing (0.0) entries stay 0.0.

    Scaling uses only the valid (non-zero) coordinate values.  The minimum
    is subtracted and then the range is divided out.  If the range is zero
    (degenerate case), the dataset is returned unchanged.

    Notes
    -----
    This mirrors the auto-scaling logic in ``bunch_of_stuff._load_posetracks``
    but preserves the missing-keypoint convention (0.0 stays 0.0).
    """
    pos = ds["position"].values.copy()  # (T, I, K, 2)

    # Build a mask of valid (non-missing) entries
    valid = (pos != 0.0).any(axis=-1)  # (T, I, K) — True when at least one coord ≠ 0

    valid_x = valid.copy()
    valid_y = valid.copy()

    pos_x = pos[..., 0]  # (T, I, K)
    pos_y = pos[..., 1]

    if valid_x.any():
        x_vals = pos_x[valid_x]
        x_min, x_max = x_vals.min(), x_vals.max()
        x_range = x_max - x_min
        if x_range > 0:
            pos_x[valid_x] = (x_vals - x_min) / x_range
        else:
            logger.warning("Autoscale: x range is 0 — skipping x normalisation.")
    else:
        logger.warning("Autoscale: no valid x values found.")

    if valid_y.any():
        y_vals = pos_y[valid_y]
        y_min, y_max = y_vals.min(), y_vals.max()
        y_range = y_max - y_min
        if y_range > 0:
            pos_y[valid_y] = (y_vals - y_min) / y_range
        else:
            logger.warning("Autoscale: y range is 0 — skipping y normalisation.")
    else:
        logger.warning("Autoscale: no valid y values found.")

    pos[..., 0] = pos_x
    pos[..., 1] = pos_y

    logger.debug("Autoscale applied.")
    return ds.assign(position=(ds["position"].dims, pos))


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def _compute_kp_coverage(ds: xr.Dataset) -> dict[str, dict[str, float]]:
    """Compute per-individual, per-keypoint visibility coverage.

    A keypoint is considered *visible* in a frame when its position is not
    ``[0.0, 0.0]`` (i.e. at least one coordinate is non-zero).

    **Denominator**: frames where the individual is *present* (has at least one
    non-zero keypoint), not the total number of frames.  This makes before/after
    comparisons fair: an individual being absent from some frames does not
    artificially deflate the keypoint coverage of the frames where they do
    appear.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{individual_name: {keypoint_name: coverage_pct (0–100)}}``.
        Coverage is ``NaN`` when the individual is never present.
    """
    pos = ds["position"].values  # (T, I, K, 2)
    ind_names = ds.coords["individuals"].values.tolist()
    kp_names = ds.coords["keypoints"].values.tolist()

    coverage: dict[str, dict[str, float]] = {}
    for i, ind in enumerate(ind_names):
        # Frames where this individual is detected (any keypoint non-zero)
        ind_present = (np.abs(pos[:, i, :, :]).sum(axis=(1, 2)) > 0)  # (T,) bool
        n_present = int(ind_present.sum())

        kp_cov: dict[str, float] = {}
        for k, kp in enumerate(kp_names):
            kp_visible = (np.abs(pos[:, i, k, :]).sum(axis=-1) > 0)  # (T,) bool
            # Count frames where kp is visible AND individual is present
            visible_when_present = int((kp_visible & ind_present).sum())
            kp_cov[kp] = (
                100.0 * visible_when_present / n_present if n_present > 0 else float("nan")
            )
        coverage[ind] = kp_cov
    return coverage


def _log_coverage_delta(stats: CleaningStats) -> None:
    """Log the change in keypoint coverage before/after cleaning."""
    for ind in stats.kp_coverage_before:
        before = stats.kp_coverage_before[ind]
        after = stats.kp_coverage_after.get(ind, {})
        logger.info("Coverage delta for '%s':", ind)
        for kp in before:
            b = before[kp]
            a = after.get(kp, 0.0)
            logger.info("  %-25s before=%5.1f%%  after=%5.1f%%  Δ=%+.1f%%", kp, b, a, a - b)
