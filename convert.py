"""Convert mmpose COCO-style JSON to *movement* xarray.Dataset.

This module handles the first stage of the pipeline: parsing a pose-estimation
result file produced by `mmpose <https://github.com/open-mmlab/mmpose>`_ and
assembling an :class:`xarray.Dataset` in the format expected by the
``movement`` library.

Expected JSON layout
--------------------
.. code-block:: json

    {
        "meta_info": {
            "dataset_name": "coco",
            "num_keypoints": 17,
            "keypoint_id2name": {"0": "nose", "1": "left_eye", ...},
            ...
        },
        "instance_info": [
            {
                "frame_id": 0,
                "instances": [
                    {
                        "keypoints": [[x0, y0], [x1, y1], ...],
                        "keypoint_scores": [s0, s1, ...],
                        "keypoints_label": 1,
                        "bbox": [[x1, y1, x2, y2]],
                        "bbox_score": 1.0
                    }
                ]
            },
            ...
        ]
    }

``keypoints_label`` is the tracker-assigned integer identity; users map this
to a human-readable name via ``identity_map``.

Output Dataset
--------------
Dimensions: ``(time, individuals, keypoints, space)``

Variables:

* ``position``   – float32, shape ``(T, I, K, 2)``; ``[0.0, 0.0]`` encodes a
  missing / low-confidence keypoint.
* ``confidence`` – float32, shape ``(T, I, K)``; raw score from mmpose.

Attributes: ``fps``, ``source_file``, ``num_frames``.
"""

import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_json_to_dataset(
    json_path: str | Path,
    identity_map: dict[int, str],
    fps: float = 25.0,
) -> xr.Dataset:
    """Parse an mmpose COCO JSON file and return a *movement* xarray.Dataset.

    Parameters
    ----------
    json_path : str or Path
        Path to the mmpose result JSON file.
    identity_map : dict[int, str]
        Mapping from ``keypoints_label`` integer to individual name.
        Example: ``{1: "child", 2: "clinician", 3: "parent"}``.
        Labels not present in this dict are **ignored**.
    fps : float, optional
        Frames per second of the source video, stored as a Dataset attribute.
        Default is 25.

    Returns
    -------
    xarray.Dataset
        Dataset with variables ``position`` and ``confidence`` and dimensions
        ``(time, individuals, keypoints, space)``.

    Raises
    ------
    ValueError
        If the JSON is missing expected keys or ``identity_map`` is empty.
    FileNotFoundError
        If ``json_path`` does not exist.

    Notes
    -----
    * Frames absent from the JSON (sparse gaps) are filled with zeros.
    * The time coordinate is frame indices (integers starting at 0).
    * Keypoint names are taken from ``meta_info.keypoint_id2name``; they are
      lowercased and spaces replaced by underscores for consistency.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not identity_map:
        raise ValueError("identity_map must not be empty.")

    logger.info("Loading JSON from %s …", json_path)
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # ------------------------------------------------------------------
    # 1. Parse meta_info
    # ------------------------------------------------------------------
    meta = data.get("meta_info", {})
    kp_id2name_raw: dict = meta.get("keypoint_id2name", {})
    if not kp_id2name_raw:
        raise ValueError("meta_info.keypoint_id2name is missing or empty.")

    # Build ordered keypoint name list (0 → K-1)
    num_kp = int(meta.get("num_keypoints", len(kp_id2name_raw)))
    keypoint_names: list[str] = [
        _normalise_kp_name(kp_id2name_raw[str(i)]) for i in range(num_kp)
    ]
    logger.info("Keypoints (%d): %s", num_kp, keypoint_names)

    # Ordered individual names (preserving insertion order of identity_map)
    individual_names: list[str] = list(dict.fromkeys(identity_map.values()))
    # label → index in individual_names
    label_to_idx: dict[int, int] = {
        lbl: individual_names.index(name)
        for lbl, name in identity_map.items()
        if name in individual_names
    }
    num_ind = len(individual_names)
    logger.info("Individuals (%d): %s", num_ind, individual_names)
    logger.info("Label → index mapping: %s", label_to_idx)

    # ------------------------------------------------------------------
    # 2. Determine total frame count
    # ------------------------------------------------------------------
    instance_info: list[dict] = data.get("instance_info", [])
    if not instance_info:
        raise ValueError("instance_info is missing or empty.")

    # Frame IDs may not be contiguous — find the maximum
    max_frame_id: int = max(entry["frame_id"] for entry in instance_info)
    num_frames: int = max_frame_id + 1
    logger.info("Total frames: %d (max frame_id=%d)", num_frames, max_frame_id)

    # ------------------------------------------------------------------
    # 3. Allocate output arrays (zeros = missing)
    # ------------------------------------------------------------------
    position = np.zeros((num_frames, num_ind, num_kp, 2), dtype=np.float32)
    confidence = np.zeros((num_frames, num_ind, num_kp), dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Fill arrays from instance_info
    # ------------------------------------------------------------------
    frames_with_missing_ind: dict[str, int] = {n: 0 for n in individual_names}
    unknown_labels_seen: set[int] = set()

    for entry in instance_info:
        frame_id: int = entry["frame_id"]
        instances: list[dict] = entry.get("instances", [])

        # Track which individuals appear in this frame
        seen_inds: set[int] = set()

        for inst in instances:
            label: int = inst.get("keypoints_label")
            if label not in label_to_idx:
                unknown_labels_seen.add(label)
                continue

            ind_idx = label_to_idx[label]
            seen_inds.add(ind_idx)

            kps: list[list[float]] = inst["keypoints"]     # [[x,y], ...]
            scores: list[float] = inst["keypoint_scores"]   # [s, ...]

            for kp_i, ((x, y), s) in enumerate(zip(kps, scores)):
                position[frame_id, ind_idx, kp_i, 0] = x
                position[frame_id, ind_idx, kp_i, 1] = y
                confidence[frame_id, ind_idx, kp_i] = s

        # Count frames where each individual is absent
        for ind_idx, name in enumerate(individual_names):
            if ind_idx not in seen_inds:
                frames_with_missing_ind[name] += 1

    if unknown_labels_seen:
        logger.warning(
            "Ignored %d unknown keypoints_label value(s): %s. "
            "Add them to identity_map if needed.",
            len(unknown_labels_seen),
            sorted(unknown_labels_seen),
        )

    # ------------------------------------------------------------------
    # 5. Log per-individual absence statistics
    # ------------------------------------------------------------------
    for name, missing_count in frames_with_missing_ind.items():
        pct = 100.0 * missing_count / num_frames if num_frames > 0 else 0.0
        logger.info(
            "Individual '%s': absent in %d / %d frames (%.1f%%)",
            name, missing_count, num_frames, pct,
        )

    # ------------------------------------------------------------------
    # 6. Build xarray.Dataset
    # ------------------------------------------------------------------
    ds = xr.Dataset(
        {
            "position": (
                ("time", "individuals", "keypoints", "space"),
                position,
            ),
            "confidence": (
                ("time", "individuals", "keypoints"),
                confidence,
            ),
        },
        coords={
            "time": np.arange(num_frames, dtype=np.int64),
            "individuals": individual_names,
            "keypoints": keypoint_names,
            "space": ["x", "y"],
        },
        attrs={
            "fps": fps,
            "source_file": str(json_path),
            "num_frames": num_frames,
            "num_individuals": num_ind,
            "num_keypoints": num_kp,
        },
    )

    logger.info(
        "Dataset assembled: shape position=%s, dtype=%s",
        position.shape, position.dtype,
    )
    return ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_identity_map(identity_map_str: str) -> dict[int, str]:
    """Parse an identity-map string into a ``{int_label: name}`` dict.

    Parameters
    ----------
    identity_map_str : str
        Comma-separated ``label:name`` pairs, e.g.
        ``"1:child,2:clinician,3:parent"``.

    Returns
    -------
    dict[int, str]

    Raises
    ------
    ValueError
        If the string is malformed.

    Examples
    --------
    >>> parse_identity_map("1:child,2:clinician")
    {1: 'child', 2: 'clinician'}
    """
    result: dict[int, str] = {}
    for part in identity_map_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid identity-map entry '{part}'. "
                "Expected format 'label:name' (e.g. '1:child')."
            )
        label_str, name = part.split(":", 1)
        try:
            label = int(label_str.strip())
        except ValueError:
            raise ValueError(
                f"Identity label '{label_str}' is not an integer."
            ) from None
        result[label] = name.strip()
    if not result:
        raise ValueError(
            "identity_map_str produced an empty mapping. "
            "Provide at least one 'label:name' pair."
        )
    return result


def _normalise_kp_name(name: str) -> str:
    """Lowercase and replace spaces/hyphens with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")
