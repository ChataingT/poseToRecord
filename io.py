"""Self-contained I/O utilities for poseToRecord.

This module provides the ``Record`` dataclass and ``dump_records`` function,
duplicated from the project's ``bunch_of_stuff.py`` so that ``poseToRecord``
is fully independent of external project modules (only the ``movement``-
compatible xarray format is shared).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import xarray as xr
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Record:
    """Data structure representing a single pose-tracking record.

    Parameters
    ----------
    id : str
        Unique identifier for the record, typically derived from the relative
        path (e.g. ``"session1/seg_001"``).
    posetracks : xarray.Dataset
        Pose-tracking data in *movement* format.  Must contain at least a
        ``position`` variable with dimensions
        ``(time, individuals, keypoints, space)``.
    annotations : xarray.Dataset or None, optional
        Annotations associated with the record, if available.
    """

    id: str
    posetracks: xr.Dataset
    annotations: xr.Dataset | None = None


def dump_records(data_path: str | Path, records: list[Record]) -> None:
    """Save a list of :class:`Record` objects to disk in *movement* format.

    Each record is stored in a sub-directory named after its ``id``.  The
    pose-tracking data is saved as ``tracking.nc`` (NetCDF, scipy engine) so
    that it can later be discovered and loaded by ``load_records()`` from
    ``bunch_of_stuff.py``.

    Parameters
    ----------
    data_path : str or Path
        Root directory where records will be written.  Created if it does
        not exist.
    records : list of Record
        Records to persist.  An empty list is silently ignored.

    Notes
    -----
    * Existing files are overwritten without warning.
    * Annotations, if present, are saved as ``annotations.nc`` alongside
      ``tracking.nc``.
    """
    data_path = Path(data_path)
    logger.info("Dumping %d record(s) to %s", len(records), data_path)

    for rec in tqdm(records, desc="Saving records"):
        rec_path = data_path / rec.id
        rec_path.mkdir(parents=True, exist_ok=True)

        tracking_path = rec_path / "tracking.nc"
        rec.posetracks.to_netcdf(tracking_path, engine="scipy")
        logger.debug("Saved posetracks → %s", tracking_path)

        if rec.annotations is not None:
            ann_path = rec_path / "annotations.nc"
            rec.annotations.to_netcdf(ann_path, engine="scipy")
            logger.debug("Saved annotations → %s", ann_path)

    logger.info("Done saving records.")
