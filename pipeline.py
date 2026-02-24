"""Orchestration pipeline and CLI entry point for poseToRecord.

This module ties together the conversion, filtering, metric computation, and
report generation steps into a single ``run_pipeline`` function that can be
called programmatically or via the command line.

Command-line usage
------------------
.. code-block:: bash

    python -m poseToRecord.pipeline \\
        --input  results_skeleton_8090_T2a_ADOS.json \\
        --output output_records/ \\
        --identity-map "1:child,2:clinician,3:parent" \\
        --dyadic-individuals child clinician \\
        --conf-threshold 0.3 \\
        --min-valid-kp 5 \\
        --min-segment-frames 300 \\
        --fps 20 \\
        --log-level INFO

Programmatic usage
------------------
.. code-block:: python

    from poseToRecord.pipeline import run_pipeline

    run_pipeline(
        input_json="results_skeleton.json",
        output_dir="output/",
        identity_map={1: "child", 2: "clinician"},
        dyadic_individuals=["child", "clinician"],
        fps=20.0,
    )
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .convert import convert_json_to_dataset, parse_identity_map
from .filter import apply_cleaning_pipeline
from .io import Record, dump_records
from .metrics import compute_all_metrics
from .report import (
    generate_html_report,
    save_segment_csvs,
    save_segment_plots,
    save_video_csvs,
    save_video_plots,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    input_json: str | Path,
    output_dir: str | Path,
    identity_map: dict[int, str],
    dyadic_individuals: list[str] | None = None,
    fps: float = 25.0,
    conf_threshold: float = 0.3,
    min_valid_kp: int = 5,
    min_segment_frames: int = 300,
    congruent_window: int = 60,
    skip_metrics: bool = False,
    skip_report: bool = False,
) -> dict:
    """Run the full poseToRecord pipeline on a single JSON file.

    Parameters
    ----------
    input_json : str or Path
        Path to the mmpose COCO-style JSON file.
    output_dir : str or Path
        Root directory where all outputs will be written.
    identity_map : dict[int, str]
        Mapping from ``keypoints_label`` integer to individual name.
        Example: ``{1: "child", 2: "clinician", 3: "parent"}``.
    dyadic_individuals : list[str] or None, optional
        The two individual names that must co-occur for a frame to be valid.
        Defaults to the first two values in ``identity_map``.
    fps : float, optional
        Frames per second of the source video.  Default 25.
    conf_threshold : float, optional
        Confidence score threshold below which keypoints are zeroed out.
        Default 0.3.
    min_valid_kp : int, optional
        Minimum number of above-threshold keypoints required per dyadic
        individual per frame.  Default 5.
    min_segment_frames : int, optional
        Minimum number of consecutive valid frames for a segment to be kept.
        Default 300 (~15 s at 20 fps).
    congruent_window : int, optional
        Rolling window (frames) for congruent-motion correlation.  Default 60.
    skip_metrics : bool, optional
        If True, skip metric computation and report generation.
    skip_report : bool, optional
        If True, compute metrics (CSV + plots) but do not generate HTML report.

    Returns
    -------
    dict
        Summary dict with keys:
        ``"video_name"``, ``"total_frames"``, ``"segments_kept"``,
        ``"total_valid_frames"``, ``"output_dir"``.

    Raises
    ------
    FileNotFoundError
        If ``input_json`` does not exist.
    ValueError
        If identity_map or dyadic_individuals are invalid.
    """
    t0 = time.monotonic()
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    video_name = input_json.stem
    video_dir = output_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    # Always write logs to a file alongside the output
    _log_fh = _attach_file_logger(video_dir / "pipeline.log")

    logger.info("=" * 70)
    logger.info("poseToRecord pipeline — %s", video_name)
    logger.info("=" * 70)
    logger.info("Input :  %s", input_json)
    logger.info("Output:  %s", video_dir)
    logger.info("FPS   :  %s", fps)
    logger.info("Identity map: %s", identity_map)

    # Default dyadic individuals: first two names
    if dyadic_individuals is None:
        dyadic_individuals = list(dict.fromkeys(identity_map.values()))[:2]
        logger.info("Dyadic individuals (auto): %s", dyadic_individuals)
    else:
        logger.info("Dyadic individuals: %s", dyadic_individuals)

    # ==================================================================
    # Step 1 — Convert JSON to xarray.Dataset
    # ==================================================================
    logger.info("--- Step 1: Converting JSON ---")
    raw_ds = convert_json_to_dataset(input_json, identity_map, fps=fps)
    logger.info(
        "Raw dataset: %d frames, %d individuals, %d keypoints",
        raw_ds.sizes["time"], raw_ds.sizes["individuals"], raw_ds.sizes["keypoints"],
    )

    # ==================================================================
    # Step 2 — Apply cleaning pipeline
    # ==================================================================
    logger.info("--- Step 2: Cleaning pipeline ---")
    full_ds, segment_datasets, stats = apply_cleaning_pipeline(
        raw_ds,
        dyadic_individuals=dyadic_individuals,
        conf_threshold=conf_threshold,
        min_valid_kp=min_valid_kp,
        min_segment_frames=min_segment_frames,
    )

    if not segment_datasets:
        logger.warning(
            "No valid segments found after cleaning. "
            "Consider relaxing conf_threshold, min_valid_kp, or min_segment_frames."
        )

    # ==================================================================
    # Step 3 — Save full record
    # ==================================================================
    logger.info("--- Step 3: Saving full record ---")
    full_ds.attrs["fps"] = fps
    full_record = Record(id=video_name, posetracks=full_ds)
    dump_records(output_dir, [full_record])
    logger.info("Full record saved: %s/tracking.nc", video_dir)

    # ==================================================================
    # Step 4 — Save segment records
    # ==================================================================
    logger.info("--- Step 4: Saving %d segment record(s) ---", len(segment_datasets))
    seg_records = []
    for i, seg_ds in enumerate(segment_datasets):
        seg_ds.attrs["fps"] = fps
        seg_id = f"{video_name}/segments/seg_{i+1:03d}"
        seg_records.append(Record(id=seg_id, posetracks=seg_ds))
    if seg_records:
        dump_records(output_dir, seg_records)
        logger.info("Segment records saved.")

    # ==================================================================
    # Step 5 — Compute metrics
    # ==================================================================
    all_raw_dfs = []
    all_norm_dfs = []

    if not skip_metrics and segment_datasets:
        logger.info("--- Step 5: Computing metrics ---")
        for i, (seg_ds, seg_rec) in enumerate(zip(segment_datasets, seg_records)):
            seg_label = f"seg_{i+1:03d}"
            logger.info("  Metrics for %s …", seg_label)
            try:
                metric_dfs = compute_all_metrics(
                    seg_ds,
                    dyadic_individuals=dyadic_individuals,
                    congruent_window=congruent_window,
                )
            except Exception as exc:
                logger.error("  Failed to compute metrics for %s: %s", seg_label, exc)
                continue

            raw_df = metric_dfs["raw"]
            norm_df = metric_dfs["normalised"]
            all_raw_dfs.append(raw_df)
            all_norm_dfs.append(norm_df)

            # Save per-segment outputs
            seg_dir = output_dir / seg_rec.id
            save_segment_csvs(seg_dir, raw_df, norm_df)
            save_segment_plots(seg_dir, raw_df, norm_df)
            logger.info("  Segment %s done.", seg_label)

        # Video-level aggregated outputs
        if all_raw_dfs:
            logger.info("  Saving video-level aggregated outputs …")
            save_video_csvs(video_dir, all_raw_dfs, all_norm_dfs, stats)
            save_video_plots(video_dir, all_raw_dfs, all_norm_dfs, stats=stats, fps=fps)
    elif skip_metrics:
        logger.info("--- Step 5: Skipping metrics (--no-metrics) ---")
    else:
        logger.info("--- Step 5: No segments to compute metrics on ---")

    # ==================================================================
    # Step 6 — HTML report
    # ==================================================================
    if not skip_report and not skip_metrics and all_raw_dfs:
        logger.info("--- Step 6: Generating HTML report ---")
        generate_html_report(video_name, video_dir, stats, all_raw_dfs, fps)
    elif skip_report:
        logger.info("--- Step 6: Skipping report (--no-report) ---")
    else:
        logger.info("--- Step 6: No data for report ---")

    # ==================================================================
    # Done
    # ==================================================================
    elapsed = time.monotonic() - t0
    logger.info("=" * 70)
    logger.info("Pipeline complete in %.1f s", elapsed)
    logger.info(
        "Summary: %d segments kept, %d total valid frames",
        stats.segments_kept, stats.frames_after_continuity,
    )
    logger.info("Output directory: %s", video_dir)
    logger.info("=" * 70)

    _detach_file_logger(_log_fh)

    return {
        "video_name": video_name,
        "total_frames": stats.total_frames,
        "segments_kept": stats.segments_kept,
        "segment_lengths": stats.segment_lengths,
        "total_valid_frames": stats.frames_after_continuity,
        "output_dir": str(video_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m poseToRecord.pipeline",
        description=(
            "Convert an mmpose COCO-style JSON file into movement-format records "
            "and compute dyadic quality metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True, type=Path,
        metavar="JSON",
        help="Path to the mmpose result JSON file.",
    )
    p.add_argument(
        "--output", "-o", required=True, type=Path,
        metavar="DIR",
        help="Output root directory.",
    )
    p.add_argument(
        "--identity-map", required=True,
        metavar="MAP",
        help=(
            "Comma-separated label:name pairs, e.g. '1:child,2:clinician,3:parent'. "
            "Labels are the keypoints_label integers in the JSON."
        ),
    )
    p.add_argument(
        "--dyadic-individuals", nargs=2, metavar=("IND_A", "IND_B"),
        default=None,
        help=(
            "Names of the two dyadic individuals. Must match names in --identity-map. "
            "Defaults to the first two names in --identity-map."
        ),
    )
    p.add_argument(
        "--fps", type=float, default=25.0,
        help="Frames per second of the source video.",
    )
    p.add_argument(
        "--conf-threshold", type=float, default=0.3,
        help="Confidence score threshold for keypoint filtering.",
    )
    p.add_argument(
        "--min-valid-kp", type=int, default=5,
        help="Min keypoints above threshold per dyadic individual per frame.",
    )
    p.add_argument(
        "--min-segment-frames", type=int, default=300,
        help="Minimum continuous valid frames for a segment to be kept.",
    )
    p.add_argument(
        "--congruent-window", type=int, default=60,
        help="Rolling window (frames) for congruent-motion correlation.",
    )
    p.add_argument(
        "--no-metrics", action="store_true",
        help="Skip metric computation entirely (only convert and filter).",
    )
    p.add_argument(
        "--no-report", action="store_true",
        help="Skip HTML report generation (still outputs CSVs and plots).",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level.",
    )
    p.add_argument(
        "--log-file", type=Path, default=None,
        help="Optional path to write logs to a file.",
    )
    return p


def _attach_file_logger(log_path: Path) -> logging.FileHandler:
    """Add a FileHandler to the poseToRecord package logger.

    The handler writes to ``log_path`` so that all pipeline logs are also
    persisted to disk alongside the output files.  The caller is responsible
    for removing the handler when the pipeline finishes (to avoid duplicate
    entries if the function is called multiple times).

    Parameters
    ----------
    log_path : Path
        Destination log file (created / appended to).

    Returns
    -------
    logging.FileHandler
        The attached handler — pass it back to :func:`_detach_file_logger`
        when done.
    """
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger("poseToRecord").addHandler(fh)
    logger.info("Log file: %s", log_path)
    return fh


def _detach_file_logger(fh: logging.FileHandler) -> None:
    """Remove and close a FileHandler previously added by :func:`_attach_file_logger`."""
    logging.getLogger("poseToRecord").removeHandler(fh)
    fh.close()


def _setup_logging(level_str: str, log_file: Path | None) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    # Parse identity map
    try:
        identity_map = parse_identity_map(args.identity_map)
    except ValueError as exc:
        parser.error(str(exc))

    run_pipeline(
        input_json=args.input,
        output_dir=args.output,
        identity_map=identity_map,
        dyadic_individuals=args.dyadic_individuals,
        fps=args.fps,
        conf_threshold=args.conf_threshold,
        min_valid_kp=args.min_valid_kp,
        min_segment_frames=args.min_segment_frames,
        congruent_window=args.congruent_window,
        skip_metrics=args.no_metrics,
        skip_report=args.no_report,
    )


if __name__ == "__main__":
    main()
