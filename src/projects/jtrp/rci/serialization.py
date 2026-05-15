import csv
import json
import os
import typing

from absl import logging

from src.projects.jtrp.rci import structure

# NOTE: Default column names for the INDOT "simple" GCP CSV file format.
# TODO: If we need to support more formats, consider a more flexible parsing
# system (e.g., user-provided column mapping functions or a plugin system).
_INDOT_NORTHING_COL = "Northing (ft (US))"
_INDOT_EASTING_COL = "Easting (ft (US))"
_INDOT_POINT_COL = "Point Number"


def save_trajectories_csv(
    trajectory_set: structure.TrajectorySet,
    output_path: str,
) -> None:
    r"""Exports and saves trajectory data as a flat ``.csv`` file.

    .. note::

        In the generated csv file, each row represents one detection. The
        columns are
        - ``frame_index``
        - ``track_id``
        - ``x1``
        - ``y1``
        - ``x2``
        - ``y2``
        - ``cx``
        - ``cy``
        - ``class_id``
        - ``class_name``
        - ``confidence``
        - ``world_x`` (empty if no georeferencing was applied)
        - ``world_y`` (empty if no georeferencing was applied)

    Args:
        trajectory_set (TrajectorySet): The trajectory set to serialize.
        output_path (str): Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "frame_index",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "class_id",
        "class_name",
        "confidence",
        "world_x",
        "world_y",
    ]

    all_detections: typing.List[typing.Dict[str, typing.Any]] = []
    for traj in trajectory_set.trajectories.values():
        for det in traj.detections:
            cx, cy = det.bbox.center
            all_detections.append(
                {
                    "frame_index": det.frame_index,
                    "track_id": det.track_id,
                    "x1": f"{det.bbox.x1:.2f}",
                    "y1": f"{det.bbox.y1:.2f}",
                    "x2": f"{det.bbox.x2:.2f}",
                    "y2": f"{det.bbox.y2:.2f}",
                    "cx": f"{cx:.2f}",
                    "cy": f"{cy:.2f}",
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": f"{det.confidence:.4f}",
                    "world_x": (
                        f"{det.world_x:.3f}" if det.world_x is not None else ""
                    ),
                    "world_y": (
                        f"{det.world_y:.3f}" if det.world_y is not None else ""
                    ),
                }
            )

    # Sort by frame_index then track_id for deterministic output.
    all_detections.sort(
        key=lambda d: (int(d["frame_index"]), int(d["track_id"])),
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_detections)

    logging.info(
        "Saved %d detections to %s",
        len(all_detections),
        output_path,
    )


def save_trajectories_json(
    trajectory_set: structure.TrajectorySet,
    output_path: str,
) -> None:
    r"""Exports and saves trajectory data as a hierarchical ``.json`` file.

    .. note::
        The output contains a metadata block with video properties and a
        trajectories block mapping track IDs to their detection sequences.

    Args:
        trajectory_set (TrajectorySet): The trajectory set to serialize.
        output_path (str): Path to the output JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    data: typing.Dict[str, typing.Any] = {
        "metadata": {
            "source_video": trajectory_set.source_video,
            "frame_width": trajectory_set.frame_width,
            "frame_height": trajectory_set.frame_height,
            "fps": trajectory_set.fps,
            "total_frames": trajectory_set.total_frames,
            "num_trajectories": len(trajectory_set.trajectories),
        },
        "trajectories": {},
    }

    for tid, traj in trajectory_set.trajectories.items():
        data["trajectories"][str(tid)] = {
            "track_id": traj.track_id,
            "dominant_class": traj.dominant_class,
            "num_detections": len(traj.detections),
            "detections": [
                {
                    "frame_index": d.frame_index,
                    "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                    "center": list(d.bbox.center),
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "world_x": d.world_x,
                    "world_y": d.world_y,
                }
                for d in traj.detections
            ],
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logging.info(
        "Saved %d trajectories to %s",
        len(trajectory_set.trajectories),
        output_path,
    )


# ---------------------------------------------------------------------------
# Camera parameters (``CameraParameters`` <-> JSON).
# ---------------------------------------------------------------------------


def save_camera_parameters(
    params: structure.CameraParameters,
    path: str,
) -> None:
    r"""Saves ``structure.CameraParameters`` to a JSON file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params.to_dict(), f, indent=2)


def load_camera_parameters(path: str) -> structure.CameraParameters:
    r"""Loads ``structure.CameraParameters`` from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return structure.CameraParameters.from_dict(data)


# ---------------------------------------------------------------------------
# Ground control points (``GroundControlPoint`` <-> JSON / CSV).
# ---------------------------------------------------------------------------


def save_gcps_to_json(
    gcps: typing.Sequence[structure.GroundControlPoint],
    path: str,
    metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> None:
    r"""Saves a sequence of GCPs to a JSON file with optional metadata."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload: typing.Dict[str, typing.Any] = {
        "gcps": [g.to_dict() for g in gcps],
    }
    if metadata is not None:
        payload["metadata"] = metadata
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_gcps_from_json(
    path: str,
) -> typing.List[structure.GroundControlPoint]:
    r"""Loads GCPs from a JSON file produced by ``save_gcps_to_json``.

    The file must contain a top-level ``gcps`` list, each entry with the
    fields documented on ``structure.GroundControlPoint``.
    """
    with open(path) as f:
        data = json.load(f)
    if "gcps" not in data:
        raise ValueError(f"JSON at {path} has no top-level 'gcps' field.")
    return [structure.GroundControlPoint.from_dict(g) for g in data["gcps"]]


def save_image_uv_csv(
    markers: typing.Sequence[typing.Tuple[str, float, float, float]],
    output_path: str,
) -> None:
    r"""Writes a ``label,u,v,area_px`` CSV.

    Each tuple is ``(label, u, v, area_px)``. The CSV format is consumed by
    ``load_image_uv_csv`` and by the GCP-assembly workflow.
    """
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "u", "v", "area_px"])
        for label, u, v, area in markers:
            writer.writerow([label, f"{u:.2f}", f"{v:.2f}", f"{area:.1f}"])


def load_image_uv_csv(
    path: str,
) -> typing.Dict[str, typing.Tuple[float, float]]:
    r"""Loads a ``label,u,v`` CSV into a ``{label: (u, v)}`` map."""
    mapping: typing.Dict[str, typing.Tuple[float, float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"label", "u", "v"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                "image_uv_csv must have columns label,u,v; got "
                f"{reader.fieldnames}."
            )
        for row in reader:
            mapping[row["label"].strip()] = (
                float(row["u"]),
                float(row["v"]),
            )
    return mapping


# ---------------------------------------------------------------------------
# Georeference (``GeoReference`` <-> JSON).
# ---------------------------------------------------------------------------


def save_georeference(geo: structure.GeoReference, path: str) -> None:
    r"""Saves a ``structure.GeoReference`` to a JSON file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(geo.to_dict(), f, indent=2)


def load_georeference(path: str) -> structure.GeoReference:
    r"""Loads a ``structure.GeoReference`` from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return structure.GeoReference.from_dict(data)


# ---------------------------------------------------------------------------
# Region of interest (``RegionOfInterest`` <-> JSON).
# ---------------------------------------------------------------------------


def save_roi_to_json(
    roi: structure.RegionOfInterest,
    path: str,
) -> None:
    r"""Saves a ``RegionOfInterest`` polygon to a JSON file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(roi.to_dict(), f, indent=2)


def load_roi_from_json(path: str) -> structure.RegionOfInterest:
    r"""Loads a ``RegionOfInterest`` polygon from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return structure.RegionOfInterest.from_dict(data)


# ---------------------------------------------------------------------------
# INDOT "simple" GCP CSV (world coordinates).
# ---------------------------------------------------------------------------


def load_world_csv(
    path: str,
    table_index: int = 1,
    northing_col: str = _INDOT_NORTHING_COL,
    easting_col: str = _INDOT_EASTING_COL,
    point_col: str = _INDOT_POINT_COL,
) -> typing.List[typing.Tuple[str, float, float]]:
    r"""Loads world-plane GCP coordinates from an INDOT "simple" CSV.

    The INDOT CSV format may concatenate multiple flight tables into a
    single file (header rows repeat). ``table_index`` selects which table
    to read (1-based; defaults to the first table).

    Args:
        path (str): Path to the CSV file.
        table_index (int): 1-based index of the GCP table to read.
        northing_col (str): Header label for the Northing column.
        easting_col (str): Header label for the Easting column.
        point_col (str): Header label for the GCP-label column.

    Returns:
        List of ``(label, world_x_easting, world_y_northing)`` tuples.

    Raises:
        ValueError: If ``table_index`` is not 1-based, the requested table
            does not exist, or no rows are parsed.
    """
    if table_index < 1:
        raise ValueError(
            f"table_index must be 1-based and positive; got {table_index}."
        )

    tables: typing.List[typing.List[typing.Tuple[str, float, float]]] = []
    current: typing.List[typing.Tuple[str, float, float]] = []
    header: typing.Optional[typing.List[str]] = None
    with open(path, newline="") as f:
        for raw in csv.reader(f):
            if not raw or all(c.strip() == "" for c in raw):
                continue
            if point_col in raw and northing_col in raw and easting_col in raw:
                if current:
                    tables.append(current)
                    current = []
                header = raw
                continue
            if header is None:
                continue
            entry = dict(zip(header, raw))
            try:
                current.append(
                    (
                        str(entry[point_col]).strip(),
                        float(entry[easting_col]),
                        float(entry[northing_col]),
                    )
                )
            except (KeyError, ValueError):
                # Malformed rows (often the second-table header in stacked
                # CSVs) are silently skipped.
                continue
    if current:
        tables.append(current)
    if not tables:
        raise ValueError(
            f"No GCP rows parsed from {path}; check header names."
        )
    if table_index > len(tables):
        raise ValueError(
            f"Requested table {table_index} but file has only "
            f"{len(tables)} table(s)."
        )
    return tables[table_index - 1]


# ---------------------------------------------------------------------------
# Trajectory data (``TrajectorySet`` <-> JSON).
# ---------------------------------------------------------------------------


def load_trajectories_json(path: str) -> structure.TrajectorySet:
    r"""Loads trajectory data from a ``.json`` file.

    Args:
        path (str): Path to the JSON file previously created by
            save_trajectories_json.

    Returns:
        Reconstructed ``TrajectorySet`` instance.
    """
    with open(path) as f:
        data = json.load(f)

    meta = data["metadata"]
    tset = structure.TrajectorySet(
        source_video=meta["source_video"],
        frame_width=meta["frame_width"],
        frame_height=meta["frame_height"],
        fps=meta["fps"],
        total_frames=meta["total_frames"],
    )

    for tid_str, traj_data in data["trajectories"].items():
        for det_data in traj_data["detections"]:
            bbox_vals = det_data["bbox"]
            detection = structure.Detection(
                frame_index=det_data["frame_index"],
                track_id=int(tid_str),
                bbox=structure.BoundingBox(
                    x1=bbox_vals[0],
                    y1=bbox_vals[1],
                    x2=bbox_vals[2],
                    y2=bbox_vals[3],
                ),
                class_id=det_data["class_id"],
                class_name=det_data["class_name"],
                confidence=det_data["confidence"],
                world_x=det_data.get("world_x"),
                world_y=det_data.get("world_y"),
            )
            tset.add_detection(detection)

    return tset
