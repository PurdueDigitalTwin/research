r"""Export ``TrajectorySet`` to a NGSIM-compatible CSV.

The Next Generation Simulation (NGSIM) trajectory data
(https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/)
publishes per-vehicle, per-frame records with the columns documented in
the NGSIM "I-80 Trajectory Data" data dictionary. The fields most
commonly referenced downstream are:

- ``Vehicle_ID`` stable track identifier
- ``Frame_ID`` 1-based frame index
- ``Total_Frames`` number of frames the vehicle appears in
- ``Global_Time`` milliseconds since the Unix epoch, or since
  trajectory start when no absolute clock is available
- ``Local_X`` / ``Local_Y`` planar coordinates relative to the study
  area's local origin
- ``Global_X`` / ``Global_Y`` planar coordinates in the world CRS
- ``v_Length`` / ``v_Width`` vehicle dimensions (units consistent with
  ``Local_*``/``Global_*``)
- ``v_Class`` integer class (NGSIM original: 1=motorcycle, 2=auto,
  3=truck; we additionally allow 4=bus for COCO consistency)
- ``v_Vel`` / ``v_Acc`` instantaneous longitudinal speed / acceleration
- ``Location`` free-text site name

This module produces a CSV with exactly these columns. Because raw video
trajectories lack lane geometry or leader/follower context, NGSIM's
``Lane_ID``, ``Preceeding`` (sic), ``Following``, ``Space_Headway``, and
``Time_Headway`` columns are omitted. Downstream code that requires those
columns can compute them from the world-coordinate trajectories.
"""

import csv
import dataclasses
import os
import typing

from absl import logging
from numpy import typing as npt
import numpy as np

from src.projects.jtrp.rci import structure

# NGSIM v_Class mapping for COCO vehicle classes. NGSIM only documents
# {1: motorcycle, 2: auto, 3: truck}; we extend with 4=bus.
_COCO_TO_NGSIM_CLASS: typing.Dict[int, int] = {
    2: 2,  # car -> auto
    3: 1,  # motorcycle -> motorcycle
    5: 4,  # bus -> bus (NGSIM extension)
    7: 3,  # truck -> truck
}

# Default physical dimensions per NGSIM v_Class, in the same world units
# as the projected trajectory (e.g., US survey feet). These match the
# NGSIM I-80 averages and are intended as plausible placeholders when the
# detector does not estimate physical size. Order: (length, width).
_DEFAULT_DIMENSIONS_FTUS: typing.Dict[int, typing.Tuple[float, float]] = {
    1: (7.5, 3.0),  # motorcycle
    2: (15.0, 6.0),  # auto
    3: (35.0, 8.0),  # truck
    4: (40.0, 8.5),  # bus
}


@dataclasses.dataclass
class NgsimExportConfig:
    r"""Configuration for the NGSIM exporter.

    Attributes:
        location (str): Free-text site name, written to the ``Location``
            column verbatim.
        local_origin (Optional[Tuple[float, float]]): Local-frame origin
            ``(x0, y0)`` subtracted from world coordinates to produce
            ``Local_X``/``Local_Y``. If ``None``, the bounding box of the
            data is used and its lower-left corner is the origin.
        global_time_zero_ms (int): The ``Global_Time`` value, in
            milliseconds, assigned to frame index 0 of the source video.
            Subsequent frames are at ``global_time_zero_ms + (frame_index
            * 1000 / fps)``. Default is ``0``.
        rotation_deg (float): Optional planar rotation, in degrees,
            applied to local coordinates so that ``Local_X`` runs along
            the traffic direction. Default is ``0`` (no rotation).
        default_dimensions (Dict[int, Tuple[float, float]]): NGSIM
            class-id → ``(length, width)`` mapping for the
            ``v_Length`` / ``v_Width`` columns. Default is
            ``_DEFAULT_DIMENSIONS_FTUS`` (units of US survey feet).
        smoothing_window (int): Half-width (in frames) of the simple
            moving-average filter applied to world positions before
            computing velocity/acceleration. ``0`` disables smoothing.
            Default is ``2`` (5-frame box filter).
        emit_filtered (bool): If ``True``, drop trajectories with fewer
            than ``min_track_length`` detections. Default is ``True``.
        min_track_length (int): Minimum number of detections required to
            export a trajectory when ``emit_filtered`` is enabled.
            Default is ``5``.
    """

    location: str
    local_origin: typing.Optional[typing.Tuple[float, float]] = None
    global_time_zero_ms: int = 0
    rotation_deg: float = 0.0
    default_dimensions: typing.Dict[
        int, typing.Tuple[float, float]
    ] = dataclasses.field(
        default_factory=lambda: dict(_DEFAULT_DIMENSIONS_FTUS)
    )
    smoothing_window: int = 2
    emit_filtered: bool = True
    min_track_length: int = 5


NGSIM_FIELDNAMES: typing.Tuple[str, ...] = (
    "Vehicle_ID",
    "Frame_ID",
    "Total_Frames",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "Global_X",
    "Global_Y",
    "v_Length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Location",
)


def _coco_class_to_ngsim(class_id: int) -> int:
    r"""Returns the NGSIM vehicle class corresponding to a COCO class ID."""
    return _COCO_TO_NGSIM_CLASS.get(int(class_id), 2)


def _moving_average(values: npt.NDArray, half_window: int) -> npt.NDArray:
    r"""Applies a symmetric moving-average filter along the first axis.

    .. note::

        Boundary handling is *edge-clamp*: values near the edges average over
        the available samples instead of being padded with zeros, so the output
        has the same length as the input.

    Args:
        values (NDArray): Input array of shape ``(N, D)``.
        half_window (int): Number of frames on either side of the center
            frame to include in the average. The total window size is
            ``(2 * half_window + 1)``. If ``0``, no smoothing is applied and the input is returned as-is.

    Returns:
        NDArray: Smoothed array of shape ``(N, D)``.
    """
    if half_window <= 0 or values.shape[0] <= 1:
        return values.astype(np.float64, copy=True)
    n = values.shape[0]
    out = np.zeros_like(values, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        out[i] = values[lo:hi].mean(axis=0)
    return out


def _compute_velocity_acceleration(
    positions: np.ndarray,
    times_s: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    r"""Computes signed speed (magnitude of velocity) and acceleration.

    Uses centered finite differences (forward/backward at the endpoints).
    Returns arrays of shape ``(N,)`` aligned with the input positions.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"``positions`` must be (N, 2); got {positions.shape}."
        )
    n = positions.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)

    # Velocity vectors via gradient (handles endpoints correctly).
    vx = np.gradient(positions[:, 0], times_s, edge_order=1)
    vy = np.gradient(positions[:, 1], times_s, edge_order=1)
    speed = np.sqrt(vx * vx + vy * vy)
    # Acceleration is the time derivative of the speed magnitude (NGSIM
    # convention: longitudinal scalar, not the vector norm of dV/dt).
    accel = np.gradient(speed, times_s, edge_order=1)
    return speed, accel


def _resolve_local_origin(
    trajectory_set: structure.TrajectorySet,
    config: NgsimExportConfig,
) -> typing.Tuple[float, float]:
    r"""Resolves the local origin for the export.

    Args:
        trajectory_set (TrajectorySet): The trajectory set being exported.
        config (NgsimExportConfig): The export configuration.

    Returns:
        A tuple of the local origin ``(x0, y0)``.
    """
    if config.local_origin is not None:
        return config.local_origin
    xs: typing.List[float] = []
    ys: typing.List[float] = []
    for traj in trajectory_set.trajectories.values():
        for det in traj.detections:
            if det.world_x is not None and det.world_y is not None:
                xs.append(det.world_x)
                ys.append(det.world_y)
    if not xs:
        return (0.0, 0.0)
    return (float(min(xs)), float(min(ys)))


def _apply_rotation(
    points: np.ndarray,
    rotation_deg: float,
) -> np.ndarray:
    if rotation_deg == 0.0:
        return points
    theta = np.deg2rad(rotation_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points @ R.T


def save_ngsim_csv(
    trajectory_set: structure.TrajectorySet,
    output_path: str,
    config: NgsimExportConfig,
) -> None:
    r"""Exports a ``TrajectorySet`` instance to a NGSIM-style CSV.

    Args:
        trajectory_set (TrajectorySet): Trajectories to export. Each
            detection must have ``world_x`` and ``world_y`` populated.
        output_path (str): Destination CSV path. The parent directory is
            created if it does not exist.
        config (NgsimExportConfig): Export configuration.

    Raises:
        ValueError: If any detection lacks world coordinates or if no
            trajectories survive the ``min_track_length`` filter.
    """
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    fps = float(trajectory_set.fps) if trajectory_set.fps else 30.0
    dt_ms = 1000.0 / fps

    origin_x, origin_y = _resolve_local_origin(trajectory_set, config)

    rows: typing.List[typing.Dict[str, typing.Any]] = []
    accepted = 0
    for tid, traj in trajectory_set.trajectories.items():
        if (
            config.emit_filtered
            and len(traj.detections) < config.min_track_length
        ):
            continue

        ordered = sorted(traj.detections, key=lambda d: d.frame_index)
        if any(d.world_x is None or d.world_y is None for d in ordered):
            raise ValueError(
                f"Trajectory {tid} has detections without world "
                "coordinates; cannot export to NGSIM format."
            )

        world = np.array(
            [(d.world_x, d.world_y) for d in ordered], dtype=np.float64
        )
        smoothed = _moving_average(world, config.smoothing_window)
        frame_idxs = np.array([d.frame_index for d in ordered], dtype=np.int64)
        times_s = frame_idxs.astype(np.float64) / fps
        speeds, accels = _compute_velocity_acceleration(smoothed, times_s)

        local = smoothed - np.array([[origin_x, origin_y]], dtype=np.float64)
        local = _apply_rotation(local, config.rotation_deg)

        total_frames = len(ordered)
        accepted += 1
        for i, det in enumerate(ordered):
            ngsim_class = _coco_class_to_ngsim(det.class_id)
            length, width = config.default_dimensions.get(
                ngsim_class, _DEFAULT_DIMENSIONS_FTUS[2]
            )
            global_time_ms = config.global_time_zero_ms + int(
                round(det.frame_index * dt_ms)
            )
            rows.append(
                {
                    "Vehicle_ID": int(tid),
                    "Frame_ID": int(det.frame_index) + 1,
                    "Total_Frames": total_frames,
                    "Global_Time": global_time_ms,
                    "Local_X": f"{local[i, 0]:.3f}",
                    "Local_Y": f"{local[i, 1]:.3f}",
                    "Global_X": f"{smoothed[i, 0]:.3f}",
                    "Global_Y": f"{smoothed[i, 1]:.3f}",
                    "v_Length": f"{length:.2f}",
                    "v_Width": f"{width:.2f}",
                    "v_Class": ngsim_class,
                    "v_Vel": f"{speeds[i]:.3f}",
                    "v_Acc": f"{accels[i]:.3f}",
                    "Location": config.location,
                }
            )

    if not rows:
        raise ValueError(
            "No trajectories satisfied the export filter; check "
            "``min_track_length`` or whether world coordinates were "
            "populated."
        )

    rows.sort(key=lambda r: (int(r["Vehicle_ID"]), int(r["Frame_ID"])))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(NGSIM_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)

    logging.info(
        "Saved NGSIM-format CSV: %d rows across %d trajectories to %s",
        len(rows),
        accepted,
        output_path,
    )
