"""Quality metrics for dyadic pose-tracking data.

This module computes a comprehensive set of kinematic and dyadic quality
metrics from a *movement*-format :class:`xarray.Dataset`.

Design principles
-----------------
* **Missing keypoints** are encoded as ``[0.0, 0.0]`` (both x and y zero).
  Every computation ignores these entries and returns ``NaN`` when there is
  insufficient data to produce a meaningful result.
* **Consistent keypoint sets across frames** – when computing frame-to-frame
  displacements (speed, velocity, …), only the *intersection* of visible
  keypoints at frames *t* and *t−1* is used.  If fewer than
  ``MIN_INTERSECTION_KP`` keypoints are shared, the transition is marked as
  ``NaN``.  A flag column ``kp_set_changed`` records frames where the visible
  set differs from the previous frame.
* **Redundant calculations** – each metric is computed from multiple keypoint
  subsets (all-visible centroid, trunk-only centroid, per-keypoint) and the
  individual traces are reported side-by-side.
* **Two output variants** – *raw 2D* (pixel space) and *trunk-height
  normalised* (pseudo-3D proxy).

Trunk height definition
-----------------------
``trunk_height = AVG(d(left_shoulder, left_hip), d(right_shoulder, right_hip))``

If only one side pair is fully visible, that side's distance is used alone.
If neither pair is available, the frame gets ``NaN`` trunk height.
The resulting time series is smoothed with a rolling median (window 25 frames)
to suppress single-frame outliers.

Metric definitions
------------------
speed_centroid / speed_trunk
    Euclidean distance of the centroid between consecutive frames.  Centroid
    is computed from the intersection of visible keypoints.

velocity_centroid_x/y / velocity_trunk_x/y
    Signed displacement of the centroid in x and y.

acceleration_centroid / acceleration_trunk
    Frame-to-frame change in speed magnitude.

kinetic_energy (agitation)
    Sum of squared displacements per visible keypoint (intersection):
    ``KE(t) = Σ_{kp ∈ intersect} ‖pos(t) − pos(t-1)‖²``.
    More robust than centroid speed to centroid-shift artifacts.

agitation_global
    Mean kinetic energy across all individuals at each frame.

interpersonal_distance_centroid / _trunk
    Euclidean distance between the two dyadic individuals' centroids.

interpersonal_approach
    Frame-to-frame change in interpersonal distance (negative = approaching).

facingness
    Cosine similarity of the two torso heading vectors.
    Heading = mid_shoulders − mid_hips (points "upward" from hips to
    shoulders, approximating the front-facing direction of the torso).
    +1 = same direction (side-by-side), −1 = face-to-face, 0 = perpendicular.

congruent_motion
    Pearson correlation of the two dyadic individuals' speed_centroid time
    series, computed over a rolling window.  Measures behavioural synchrony.

speed_kp_{name}
    Per-keypoint speed (one column per keypoint, NaN when not visible at
    both consecutive frames).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Minimum number of shared visible keypoints required to compute a valid
# frame-to-frame displacement.
MIN_INTERSECTION_KP: int = 3

# COCO keypoint names used for trunk / facingness computation
_TRUNK_KPS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
_LEFT_PAIR = ("left_shoulder", "left_hip")
_RIGHT_PAIR = ("right_shoulder", "right_hip")

# Rolling window for congruent motion (frames)
DEFAULT_CONGRUENT_WINDOW: int = 60

# Rolling median window for trunk-height smoothing (frames)
TRUNK_SMOOTH_WINDOW: int = 25


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_all_metrics(
    ds: xr.Dataset,
    dyadic_individuals: list[str],
    congruent_window: int = DEFAULT_CONGRUENT_WINDOW,
) -> dict[str, pd.DataFrame]:
    """Compute all quality metrics and return them as a dict of DataFrames.

    Parameters
    ----------
    ds : xarray.Dataset
        Cleaned pose-tracking dataset (``position``, ``confidence`` variables,
        dimensions ``time × individuals × keypoints × space``).
    dyadic_individuals : list[str]
        Exactly two individual names forming the dyad.  Must be present in
        ``ds.coords["individuals"]``.
    congruent_window : int, optional
        Rolling window length (frames) for congruent-motion correlation.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys:

        * ``"raw"``        – all metrics in pixel / original units.
        * ``"normalised"`` – same metrics divided by trunk height where
          applicable (distances and speeds only; angles unchanged).

    Notes
    -----
    Both DataFrames share the same index (frame indices from ``ds.coords["time"]``).
    """
    if len(dyadic_individuals) != 2:
        raise ValueError("dyadic_individuals must contain exactly 2 names.")
    ind_a, ind_b = dyadic_individuals

    ind_names = ds.coords["individuals"].values.tolist()
    for ind in dyadic_individuals:
        if ind not in ind_names:
            raise ValueError(
                f"Individual '{ind}' not found in dataset (available: {ind_names})."
            )

    kp_names = ds.coords["keypoints"].values.tolist()
    pos = ds["position"].values  # (T, I, K, 2)
    n_frames = pos.shape[0]
    time_idx = ds.coords["time"].values

    logger.info("Computing metrics for %d frames, dyad: %s × %s", n_frames, ind_a, ind_b)

    # ------------------------------------------------------------------
    # Build per-individual position arrays
    # ------------------------------------------------------------------
    idx_a = ind_names.index(ind_a)
    idx_b = ind_names.index(ind_b)
    pos_a = pos[:, idx_a, :, :]  # (T, K, 2)
    pos_b = pos[:, idx_b, :, :]  # (T, K, 2)

    # ------------------------------------------------------------------
    # Trunk heights (per individual, smoothed)
    # ------------------------------------------------------------------
    trunk_h_a = _compute_trunk_height(pos_a, kp_names)  # (T,)
    trunk_h_b = _compute_trunk_height(pos_b, kp_names)  # (T,)

    # ------------------------------------------------------------------
    # Per-individual kinematic metrics
    # ------------------------------------------------------------------
    df_a = _individual_kinematics(pos_a, kp_names, prefix=ind_a)
    df_b = _individual_kinematics(pos_b, kp_names, prefix=ind_b)

    # ------------------------------------------------------------------
    # Agitation (kinetic energy)
    # ------------------------------------------------------------------
    ke_a = _kinetic_energy(pos_a, kp_names)  # (T,)
    ke_b = _kinetic_energy(pos_b, kp_names)  # (T,)

    # ------------------------------------------------------------------
    # Dyadic metrics
    # ------------------------------------------------------------------
    df_dyadic = _dyadic_metrics(
        pos_a, pos_b, kp_names, ind_a, ind_b,
        ke_a, ke_b, congruent_window,
    )

    # ------------------------------------------------------------------
    # Assemble raw DataFrame
    # ------------------------------------------------------------------
    raw = pd.concat([df_a, df_b, df_dyadic], axis=1)
    raw.index = time_idx
    raw.index.name = "frame"

    # Global agitation
    raw["agitation_global_ke"] = raw[[f"{ind_a}_kinetic_energy", f"{ind_b}_kinetic_energy"]].mean(axis=1)

    # ------------------------------------------------------------------
    # Normalised DataFrame (divide length/speed metrics by trunk height)
    # ------------------------------------------------------------------
    norm = raw.copy()

    dist_and_speed_cols_a = [c for c in df_a.columns if "speed" in c or "acc" in c or "ke" in c.lower()]
    dist_and_speed_cols_b = [c for c in df_b.columns if "speed" in c or "acc" in c or "ke" in c.lower()]
    dyad_dist_cols = [c for c in df_dyadic.columns if "distance" in c or "approach" in c]

    # Normalise individual A metrics by A's trunk height
    for col in dist_and_speed_cols_a:
        if col in norm.columns:
            norm[col] = _safe_divide(norm[col].values, trunk_h_a)

    # Normalise individual B metrics by B's trunk height
    for col in dist_and_speed_cols_b:
        if col in norm.columns:
            norm[col] = _safe_divide(norm[col].values, trunk_h_b)

    # Normalise dyadic distances by mean trunk height
    mean_trunk = np.where(
        np.isnan(trunk_h_a) | np.isnan(trunk_h_b),
        np.nan,
        (trunk_h_a + trunk_h_b) / 2.0,
    )
    for col in dyad_dist_cols:
        if col in norm.columns:
            norm[col] = _safe_divide(norm[col].values, mean_trunk)

    # Also normalise global agitation by mean trunk height
    norm["agitation_global_ke"] = _safe_divide(norm["agitation_global_ke"].values, mean_trunk)

    norm.index = time_idx
    norm.index.name = "frame"

    logger.info(
        "Metrics computed: %d columns (raw), %d columns (normalised)",
        len(raw.columns), len(norm.columns),
    )
    return {"raw": raw, "normalised": norm}


# ---------------------------------------------------------------------------
# Per-individual kinematics
# ---------------------------------------------------------------------------


def _individual_kinematics(pos: np.ndarray, kp_names: list[str], prefix: str) -> pd.DataFrame:
    """Compute speed, velocity, acceleration, KE for a single individual.

    Parameters
    ----------
    pos : np.ndarray, shape (T, K, 2)
        Position array for one individual.
    kp_names : list[str]
        Keypoint names in order.
    prefix : str
        Column name prefix (individual name).

    Returns
    -------
    pd.DataFrame
        Columns (all NaN-aware):
        ``{prefix}_speed_centroid``, ``{prefix}_speed_trunk``,
        ``{prefix}_velocity_centroid_x/y``, ``{prefix}_velocity_trunk_x/y``,
        ``{prefix}_acceleration_centroid``, ``{prefix}_acceleration_trunk``,
        ``{prefix}_kinetic_energy``,
        ``{prefix}_kp_set_changed``,
        ``{prefix}_speed_kp_{kp_name}`` for each keypoint.
    """
    n_frames, n_kp, _ = pos.shape
    logger.debug("Computing kinematics for '%s': %d frames, %d kps", prefix, n_frames, n_kp)

    # Visible mask: keypoint visible when at least one coordinate != 0
    visible = (np.abs(pos).sum(axis=-1) > 0)  # (T, K)  bool

    # ---- Centroid speed / velocity ----
    speed_c, vx_c, vy_c, kp_set_changed = _centroid_speed_velocity(pos, visible, "all")
    acc_c = _acceleration(speed_c)

    # ---- Trunk centroid speed / velocity ----
    trunk_kp_idx = [kp_names.index(k) for k in _TRUNK_KPS if k in kp_names]
    if trunk_kp_idx:
        pos_trunk = pos[:, trunk_kp_idx, :]
        vis_trunk = visible[:, trunk_kp_idx]
        speed_t, vx_t, vy_t, _ = _centroid_speed_velocity(pos_trunk, vis_trunk, "trunk")
        acc_t = _acceleration(speed_t)
    else:
        logger.warning("Trunk keypoints not found for '%s' — trunk metrics will be NaN.", prefix)
        speed_t = vx_t = vy_t = acc_t = np.full(n_frames, np.nan)

    # ---- Kinetic energy ----
    ke = _kinetic_energy(pos, kp_names)

    # ---- Per-keypoint speed ----
    per_kp_speed: dict[str, np.ndarray] = {}
    for k, kp in enumerate(kp_names):
        spd_kp = np.full(n_frames, np.nan)
        for t in range(1, n_frames):
            if visible[t, k] and visible[t - 1, k]:
                spd_kp[t] = float(np.linalg.norm(pos[t, k] - pos[t - 1, k]))
        per_kp_speed[kp] = spd_kp

    # ---- Assemble DataFrame ----
    df = pd.DataFrame(index=np.arange(n_frames))
    df[f"{prefix}_speed_centroid"] = speed_c
    df[f"{prefix}_speed_trunk"] = speed_t
    df[f"{prefix}_velocity_centroid_x"] = vx_c
    df[f"{prefix}_velocity_centroid_y"] = vy_c
    df[f"{prefix}_velocity_trunk_x"] = vx_t
    df[f"{prefix}_velocity_trunk_y"] = vy_t
    df[f"{prefix}_acceleration_centroid"] = acc_c
    df[f"{prefix}_acceleration_trunk"] = acc_t
    df[f"{prefix}_kinetic_energy"] = ke
    df[f"{prefix}_kp_set_changed"] = kp_set_changed
    for kp, spd in per_kp_speed.items():
        df[f"{prefix}_speed_kp_{kp}"] = spd

    return df


def _centroid_speed_velocity(
    pos: np.ndarray,
    visible: np.ndarray,
    subset_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute centroid speed and velocity using intersection of visible kps.

    Parameters
    ----------
    pos : np.ndarray, shape (T, K, 2)
    visible : np.ndarray, shape (T, K), bool
    subset_label : str
        Label for logging.

    Returns
    -------
    speed : np.ndarray, shape (T,) — NaN at t=0 and invalid transitions.
    vx : np.ndarray, shape (T,)
    vy : np.ndarray, shape (T,)
    kp_set_changed : np.ndarray, shape (T,), bool
    """
    n_frames, n_kp, _ = pos.shape
    speed = np.full(n_frames, np.nan)
    vx = np.full(n_frames, np.nan)
    vy = np.full(n_frames, np.nan)
    kp_set_changed = np.zeros(n_frames, dtype=bool)

    for t in range(1, n_frames):
        shared = visible[t] & visible[t - 1]  # (K,) bool
        n_shared = int(shared.sum())

        if n_shared < MIN_INTERSECTION_KP:
            # Not enough shared keypoints — mark as NaN
            continue

        c_curr = pos[t, shared, :].mean(axis=0)     # (2,)
        c_prev = pos[t - 1, shared, :].mean(axis=0)  # (2,)
        disp = c_curr - c_prev
        speed[t] = float(np.linalg.norm(disp))
        vx[t] = float(disp[0])
        vy[t] = float(disp[1])

        # Flag if the visible set changed vs previous frame
        if not np.array_equal(visible[t], visible[t - 1]):
            kp_set_changed[t] = True

    return speed, vx, vy, kp_set_changed


def _acceleration(speed: np.ndarray) -> np.ndarray:
    """Compute acceleration as the absolute frame-to-frame change in speed."""
    acc = np.full_like(speed, np.nan)
    for t in range(1, len(speed)):
        if not np.isnan(speed[t]) and not np.isnan(speed[t - 1]):
            acc[t] = abs(speed[t] - speed[t - 1])
    return acc


def _kinetic_energy(pos: np.ndarray, kp_names: list[str]) -> np.ndarray:
    """Compute per-frame kinetic energy proxy.

    ``KE(t) = Σ_{kp ∈ intersect(t, t-1)} ‖pos_kp(t) − pos_kp(t-1)‖²``

    The sum is over the intersection of visible keypoints at t and t−1.
    Proportional to kinetic energy under uniform mass assumption; captures
    distributed body movement better than centroid speed.

    Parameters
    ----------
    pos : np.ndarray, shape (T, K, 2)
    kp_names : list[str]
        Not used for computation but kept for API consistency.
    """
    n_frames = pos.shape[0]
    ke = np.full(n_frames, np.nan)
    visible = (np.abs(pos).sum(axis=-1) > 0)  # (T, K)

    for t in range(1, n_frames):
        shared = visible[t] & visible[t - 1]
        if not shared.any():
            continue
        disp = pos[t, shared, :] - pos[t - 1, shared, :]  # (n_shared, 2)
        ke[t] = float((disp ** 2).sum())

    return ke


# ---------------------------------------------------------------------------
# Trunk height
# ---------------------------------------------------------------------------


def compute_trunk_height(ds: xr.Dataset, individual: str) -> np.ndarray:
    """Compute trunk height for one individual in the dataset.

    Convenience wrapper around :func:`_compute_trunk_height`.

    Parameters
    ----------
    ds : xarray.Dataset
    individual : str

    Returns
    -------
    np.ndarray, shape (T,)
        Smoothed trunk height per frame (NaN where unavailable).
    """
    ind_names = ds.coords["individuals"].values.tolist()
    kp_names = ds.coords["keypoints"].values.tolist()
    idx = ind_names.index(individual)
    pos = ds["position"].values[:, idx, :, :]  # (T, K, 2)
    return _compute_trunk_height(pos, kp_names)


def _compute_trunk_height(pos: np.ndarray, kp_names: list[str]) -> np.ndarray:
    """Internal trunk height computation (raw + smoothed).

    ``trunk_height = AVG(d(left_shoulder, left_hip), d(right_shoulder, right_hip))``

    Falls back to whichever side pair is fully visible if only one is.

    Parameters
    ----------
    pos : np.ndarray, shape (T, K, 2)
    kp_names : list[str]

    Returns
    -------
    np.ndarray, shape (T,) — smoothed with rolling median.
    """
    n_frames = pos.shape[0]
    trunk_h = np.full(n_frames, np.nan)

    # Get keypoint indices (may be absent)
    def _idx(name: str) -> int | None:
        return kp_names.index(name) if name in kp_names else None

    ls_i = _idx("left_shoulder")
    rs_i = _idx("right_shoulder")
    lh_i = _idx("left_hip")
    rh_i = _idx("right_hip")

    visible = (np.abs(pos).sum(axis=-1) > 0)  # (T, K)

    for t in range(n_frames):
        distances = []

        # Left pair
        if ls_i is not None and lh_i is not None:
            if visible[t, ls_i] and visible[t, lh_i]:
                d = float(np.linalg.norm(pos[t, ls_i] - pos[t, lh_i]))
                distances.append(d)

        # Right pair
        if rs_i is not None and rh_i is not None:
            if visible[t, rs_i] and visible[t, rh_i]:
                d = float(np.linalg.norm(pos[t, rs_i] - pos[t, rh_i]))
                distances.append(d)

        if distances:
            trunk_h[t] = float(np.mean(distances))

    # Smooth with rolling median
    trunk_h_smoothed = _rolling_median(trunk_h, TRUNK_SMOOTH_WINDOW)
    logger.debug(
        "Trunk height: %.1f%% valid frames",
        100.0 * (~np.isnan(trunk_h_smoothed)).sum() / max(n_frames, 1),
    )
    return trunk_h_smoothed


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply a rolling median (NaN-aware) to a 1-D array."""
    result = np.full_like(arr, np.nan)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        vals = arr[lo:hi]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            result[i] = float(np.median(valid))
    return result


# ---------------------------------------------------------------------------
# Dyadic metrics
# ---------------------------------------------------------------------------


def _dyadic_metrics(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    kp_names: list[str],
    ind_a: str,
    ind_b: str,
    ke_a: np.ndarray,
    ke_b: np.ndarray,
    congruent_window: int,
) -> pd.DataFrame:
    """Compute all dyadic metrics between two individuals.

    Returns
    -------
    pd.DataFrame with columns for interpersonal distance, approach/retreat,
    facingness, congruent motion, and global agitation.
    """
    n_frames = pos_a.shape[0]
    df = pd.DataFrame(index=np.arange(n_frames))

    # ---- Interpersonal distance (all-visible centroid) ----
    dist_c, dist_t = _interpersonal_distances(pos_a, pos_b, kp_names)
    df["interpersonal_distance_centroid"] = dist_c
    df["interpersonal_distance_trunk"] = dist_t

    # ---- Approach / retreat ----
    approach = np.full(n_frames, np.nan)
    for t in range(1, n_frames):
        if not np.isnan(dist_c[t]) and not np.isnan(dist_c[t - 1]):
            approach[t] = dist_c[t] - dist_c[t - 1]
    df["interpersonal_approach"] = approach  # negative = approaching

    # ---- Facingness ----
    df["facingness"] = _facingness(pos_a, pos_b, kp_names)

    # ---- Congruent motion ----
    # Use centroid speed; extracted from ke as proxy — actually compute
    # properly using centroid speed from the individual kinematics calls
    # We need speed arrays: compute them here from positions
    vis_a = (np.abs(pos_a).sum(axis=-1) > 0)
    vis_b = (np.abs(pos_b).sum(axis=-1) > 0)
    speed_a, _, _, _ = _centroid_speed_velocity(pos_a, vis_a, ind_a)
    speed_b, _, _, _ = _centroid_speed_velocity(pos_b, vis_b, ind_b)
    df["congruent_motion"] = _congruent_motion(speed_a, speed_b, congruent_window)

    return df


def _interpersonal_distances(
    pos_a: np.ndarray, pos_b: np.ndarray, kp_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centroid-based and trunk-based interpersonal distance.

    Returns
    -------
    dist_centroid : np.ndarray, shape (T,)
    dist_trunk : np.ndarray, shape (T,)
    """
    n_frames = pos_a.shape[0]
    vis_a = (np.abs(pos_a).sum(axis=-1) > 0)
    vis_b = (np.abs(pos_b).sum(axis=-1) > 0)

    dist_c = np.full(n_frames, np.nan)
    dist_t = np.full(n_frames, np.nan)

    trunk_idx = [kp_names.index(k) for k in _TRUNK_KPS if k in kp_names]

    for t in range(n_frames):
        # All-visible centroid
        n_a = vis_a[t].sum()
        n_b = vis_b[t].sum()
        if n_a >= 3 and n_b >= 3:
            ca = pos_a[t, vis_a[t], :].mean(axis=0)
            cb = pos_b[t, vis_b[t], :].mean(axis=0)
            dist_c[t] = float(np.linalg.norm(ca - cb))

        # Trunk centroid
        if trunk_idx:
            vis_t_a = vis_a[t][trunk_idx]
            vis_t_b = vis_b[t][trunk_idx]
            if vis_t_a.sum() >= 2 and vis_t_b.sum() >= 2:
                pos_t_a = pos_a[t][trunk_idx][vis_t_a]
                pos_t_b = pos_b[t][trunk_idx][vis_t_b]
                ta = pos_t_a.mean(axis=0)
                tb = pos_t_b.mean(axis=0)
                dist_t[t] = float(np.linalg.norm(ta - tb))

    logger.debug(
        "Interpersonal distance: centroid valid=%.1f%%, trunk valid=%.1f%%",
        100.0 * (~np.isnan(dist_c)).mean(),
        100.0 * (~np.isnan(dist_t)).mean(),
    )
    return dist_c, dist_t


def _facingness(
    pos_a: np.ndarray, pos_b: np.ndarray, kp_names: list[str]
) -> np.ndarray:
    """Compute facingness (torso cosine similarity) per frame.

    Heading vector = mid(left_shoulder, right_shoulder) − mid(left_hip, right_hip).

    This vector points from the person's hips toward their shoulders,
    approximating the "upward / forward" torso direction in 2D.  The cosine
    similarity of heading_A and heading_B reflects their relative orientation:

    * +1  → same direction (walking side by side, parallel torsos)
    * −1  → opposite directions (face-to-face, anti-parallel torsos)
    *  0  → perpendicular orientations

    Note: in a frontal camera view, two people facing each other will have
    approximately opposite heading vectors, yielding facingness ≈ −1.

    Returns
    -------
    np.ndarray, shape (T,)
    """
    n_frames = pos_a.shape[0]
    result = np.full(n_frames, np.nan)

    def _idx(name: str) -> int | None:
        return kp_names.index(name) if name in kp_names else None

    ls_i = _idx("left_shoulder")
    rs_i = _idx("right_shoulder")
    lh_i = _idx("left_hip")
    rh_i = _idx("right_hip")

    if any(i is None for i in [ls_i, rs_i, lh_i, rh_i]):
        logger.warning("Cannot compute facingness: missing trunk keypoints in dataset.")
        return result

    vis_a = (np.abs(pos_a).sum(axis=-1) > 0)
    vis_b = (np.abs(pos_b).sum(axis=-1) > 0)

    valid_frames = 0
    for t in range(n_frames):
        # Check all 4 trunk kps visible for both individuals
        a_ok = vis_a[t, ls_i] and vis_a[t, rs_i] and vis_a[t, lh_i] and vis_a[t, rh_i]
        b_ok = vis_b[t, ls_i] and vis_b[t, rs_i] and vis_b[t, lh_i] and vis_b[t, rh_i]

        if not (a_ok and b_ok):
            continue

        # Heading vector A: mid_shoulders_A − mid_hips_A
        mid_sh_a = (pos_a[t, ls_i] + pos_a[t, rs_i]) / 2.0
        mid_hp_a = (pos_a[t, lh_i] + pos_a[t, rh_i]) / 2.0
        h_a = mid_sh_a - mid_hp_a

        # Heading vector B
        mid_sh_b = (pos_b[t, ls_i] + pos_b[t, rs_i]) / 2.0
        mid_hp_b = (pos_b[t, lh_i] + pos_b[t, rh_i]) / 2.0
        h_b = mid_sh_b - mid_hp_b

        norm_a = np.linalg.norm(h_a)
        norm_b = np.linalg.norm(h_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            continue

        result[t] = float(np.dot(h_a, h_b) / (norm_a * norm_b))
        valid_frames += 1

    logger.debug(
        "Facingness: valid in %d / %d frames (%.1f%%)",
        valid_frames, n_frames, 100.0 * valid_frames / max(n_frames, 1),
    )
    return result


def _congruent_motion(
    speed_a: np.ndarray, speed_b: np.ndarray, window: int
) -> np.ndarray:
    """Rolling Pearson correlation of two speed time series.

    Measures behavioural synchrony: whether the two individuals tend to move
    and stay still at the same times.  A high positive value indicates
    coordinated or imitative movement.  Computed over a rolling window to
    capture temporal co-variation rather than instantaneous coincidence.

    A window is considered valid when < 30 % of frames have NaN speed.

    Parameters
    ----------
    speed_a, speed_b : np.ndarray, shape (T,)
    window : int

    Returns
    -------
    np.ndarray, shape (T,)
    """
    n_frames = len(speed_a)
    result = np.full(n_frames, np.nan)
    nan_tolerance = 0.30  # max fraction of NaN frames in a window

    for t in range(window - 1, n_frames):
        wa = speed_a[t - window + 1 : t + 1]
        wb = speed_b[t - window + 1 : t + 1]

        both_valid = ~(np.isnan(wa) | np.isnan(wb))
        if both_valid.mean() < (1.0 - nan_tolerance):
            continue

        wa_v = wa[both_valid]
        wb_v = wb[both_valid]
        if len(wa_v) < 3:
            continue

        # Pearson correlation
        if wa_v.std() < 1e-9 or wb_v.std() < 1e-9:
            # Constant signal → correlation undefined
            continue

        result[t] = float(np.corrcoef(wa_v, wb_v)[0, 1])

    valid_pct = 100.0 * (~np.isnan(result)).mean()
    logger.debug(
        "Congruent motion (window=%d): valid %.1f%% of frames", window, valid_pct,
    )
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division; returns NaN where denominator is NaN or zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (denominator == 0) | np.isnan(denominator),
            np.nan,
            numerator / denominator,
        )
    return result
