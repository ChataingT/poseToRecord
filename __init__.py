"""poseToRecord — Convert mmpose pose estimation JSON to *movement* records.

This package converts mmpose COCO-style JSON output into the *movement*-format
NetCDF records loadable by ``load_records()``, and computes a comprehensive set
of dyadic quality metrics.

Public API
----------
The two main entry points are:

``run_pipeline``
    Full pipeline: JSON → filtering → NetCDF records → metrics → HTML report.

``convert_json_to_dataset``
    Lower-level: JSON → raw :class:`xarray.Dataset` only (no filtering).

Quick start
-----------
.. code-block:: python

    from poseToRecord import run_pipeline

    summary = run_pipeline(
        input_json="results_skeleton_8090_T2a_ADOS.json",
        output_dir="output/",
        identity_map={1: "child", 2: "clinician", 3: "parent"},
        dyadic_individuals=["child", "clinician"],
        fps=20.0,
        conf_threshold=0.3,
        min_valid_kp=5,
        min_segment_frames=300,
    )
    print(summary)

Command-line usage
------------------
.. code-block:: bash

    python -m poseToRecord.pipeline --help

Module overview
---------------
* :mod:`poseToRecord.convert`  — JSON → xarray.Dataset
* :mod:`poseToRecord.filter`   — Confidence / dyadic / continuity filtering
* :mod:`poseToRecord.metrics`  — Kinematic and dyadic quality metrics
* :mod:`poseToRecord.report`   — CSV, plots, and HTML report generation
* :mod:`poseToRecord.io`       — Self-contained Record dataclass + dump_records
* :mod:`poseToRecord.pipeline` — Orchestration + CLI entry point
"""

from .convert import convert_json_to_dataset, parse_identity_map
from .filter import apply_cleaning_pipeline, find_continuous_segments
from .io import Record, dump_records
from .metrics import compute_all_metrics, compute_trunk_height
from .pipeline import run_pipeline

__all__ = [
    "run_pipeline",
    "convert_json_to_dataset",
    "parse_identity_map",
    "apply_cleaning_pipeline",
    "find_continuous_segments",
    "Record",
    "dump_records",
    "compute_all_metrics",
    "compute_trunk_height",
]
