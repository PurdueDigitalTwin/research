import csv
import json
import os
import typing

from src.projects.jtrp.experimental import structure
from src.utilities import logging


def save_trajectories_csv(
    trajectory_set: structure.TrajectorySet,
    output_path: str,
) -> None:
    r"""Exports and saves trajectory data as a flat `.csv` file.

    .. note::

        In the generated csv file, each row represents one detection. The
        columns are
        - `frame_index`
        - `track_id`
        - `x1`
        - `y1`
        - `x2`
        - `y2`
        - `cx`
        - `cy`
        - `class_id`
        - `class_name`
        - `confidence`

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

    logging.rank_zero_info(
        "Saved %d detections to %s",
        len(all_detections),
        output_path,
    )


def save_trajectories_json(
    trajectory_set: structure.TrajectorySet,
    output_path: str,
) -> None:
    r"""Exports and saves trajectory data as a hierarchical `.json` file.

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
                }
                for d in traj.detections
            ],
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logging.rank_zero_info(
        "Saved %d trajectories to %s",
        len(trajectory_set.trajectories),
        output_path,
    )


def load_trajectories_json(path: str) -> structure.TrajectorySet:
    r"""Loads trajectory data from a `.json` file.

    Args:
        path (str): Path to the JSON file previously created by
            save_trajectories_json.

    Returns:
        Reconstructed `TrajectorySet` instance.
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
            )
            tset.add_detection(detection)

    return tset
