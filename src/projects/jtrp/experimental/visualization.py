import typing

import cv2
from tqdm import auto as tqdm
from tqdm.contrib import logging as tqdm_logging

from src.projects.jtrp.experimental import structure
from src.utilities import logging

# Color palette for track IDs (BGR format for OpenCV).
_COLORS: typing.List[typing.Tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
    (255, 0, 128),
    (0, 255, 128),
]


def _get_color(track_id: int) -> typing.Tuple[int, int, int]:
    """Returns a deterministic color for a given track ID."""
    return _COLORS[track_id % len(_COLORS)]


def render_annotated_video(
    trajectory_set: structure.TrajectorySet,
    output_path: str,
    trail_length: int = 30,
    show_bbox: bool = True,
    show_label: bool = True,
    show_trail: bool = True,
    codec: str = "mp4v",
) -> None:
    """Renders an annotated video with bounding boxes, IDs, and trails.

    Reads the source video frame by frame, overlays tracking annotations,
    and writes the annotated frames to a new video file.

    Args:
        trajectory_set (TrajectorySet): Trajectory data with source_video path.
        output_path (str): Path for the output annotated video.
        trail_length (int, optional): Number of past frames to draw trajectory
            trail. Default is :math:`30`.
        show_bbox (bool, optional): Whether to plot the bounding boxes.
            Default is `True`.
        show_label (bool, optional): Whether to show track ID and class label.
            Default is `True`.
        show_trail (bool, optional): Whether to draw trajectory trails.
            Default is True,
        codec (str, optional): FourCC codec string (e.g., `"mp4v"`, `"XVID"`).
            Default is `"mp4v"`.
    """
    cap = cv2.VideoCapture(trajectory_set.source_video)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {trajectory_set.source_video}")

    # NOTE: ignore the type checking here due to known issue
    # See https://github.com/opencv/opencv/issues/24818
    fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        trajectory_set.fps,
        (trajectory_set.frame_width, trajectory_set.frame_height),
    )

    # Pre-build a frame-indexed lookup: frame_idx -> list of Detection.
    frame_detections: typing.Dict[int, typing.List[structure.Detection]] = {}
    for traj in trajectory_set.trajectories.values():
        for det in traj.detections:
            frame_detections.setdefault(det.frame_index, []).append(det)

    # Accumulate trail history per track: track_id -> list of (cx, cy).
    trail_history: typing.Dict[int, typing.List[typing.Tuple[int, int]]] = {}

    frame_idx = 0
    pbar = tqdm.tqdm(
        total=trajectory_set.total_frames,
        desc="Rendering",
        position=0,
        leave=False,
        unit="frames",
    )
    with tqdm_logging.logging_redirect_tqdm():
        while cap.isOpened():
            ret, frame = cap.read()
            pbar.update(1)
            if not ret:
                break

            detections = frame_detections.get(frame_idx, [])

            for det in detections:
                color = _get_color(det.track_id)
                cx, cy = det.bbox.center
                cx_int, cy_int = int(cx), int(cy)

                # Update trail history.
                if det.track_id not in trail_history:
                    trail_history[det.track_id] = []
                trail_history[det.track_id].append((cx_int, cy_int))

                # Draw bounding box.
                if show_bbox:
                    pt1 = (int(det.bbox.x1), int(det.bbox.y1))
                    pt2 = (int(det.bbox.x2), int(det.bbox.y2))
                    cv2.rectangle(frame, pt1, pt2, color, 2)

                # Draw label.
                if show_label:
                    label = (
                        f"ID:{det.track_id} {det.class_name}"
                        f" {det.confidence:.2f}"
                    )
                    label_pos = (int(det.bbox.x1), int(det.bbox.y1) - 10)
                    cv2.putText(
                        frame,
                        label,
                        label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                # Draw trajectory trail.
                if show_trail:
                    trail = trail_history[det.track_id]
                    trail_points = trail[-trail_length:]
                    for j in range(1, len(trail_points)):
                        alpha = j / len(trail_points)
                        thickness = max(1, int(2 * alpha))
                        cv2.line(
                            frame,
                            trail_points[j - 1],
                            trail_points[j],
                            color,
                            thickness,
                        )

            writer.write(frame)
            frame_idx += 1

    pbar.close()
    cap.release()
    writer.release()
    logging.rank_zero_info("Annotated video saved to %s", output_path)  #
