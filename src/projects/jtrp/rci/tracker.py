r"""Vehicle detection + tracking with YOLO + BoT-SORT / ByteTrack.

Given a video (and optionally a camera calibration and a planar pixel↔world
georeference), this module produces a ``structure.TrajectorySet``
populated with per-detection bounding boxes, track IDs, and—if a
``georeferencing.GeoReference`` is supplied—world-plane coordinates
of the bounding-box bottom-center (the canonical ground-contact proxy for
a near-nadir aerial camera).

Two execution paths are supported:

1. **Streaming path** (no calibration): uses ultralytics' streaming reader
   for constant-memory processing of arbitrarily long videos.
2. **Frame-by-frame path** (calibration supplied): reads each frame with
   OpenCV and undistorts before feeding into ``ultralytics.YOLO.track``.
"""

import typing

import cv2
import numpy as np
import torch
from tqdm import auto as tqdm
from tqdm.contrib import logging as tqdm_logging
import ultralytics

from src.projects.jtrp.rci import calibration as calibration_lib
from src.projects.jtrp.rci import georeferencing as georef_lib
from src.projects.jtrp.rci import structure
from src.utilities import logging

# COCO vehicle class IDs used by default YOLO models.
_DEFAULT_VEHICLE_CLASSES: typing.FrozenSet[int] = frozenset(
    {
        2,  # car
        3,  # motorcycle
        5,  # bus
        7,  # truck
    }
)


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> ultralytics.YOLO:
    r"""Loads a YOLO model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to ``.pt`` weights file or a model name
            (e.g., ``"yolo11n.pt"``).
        device (str, optional): Device string (``"cpu"``, ``"cuda:0"``,
            ``"mps"``). Default is ``"cpu"``.

    Returns:
        Loaded YOLO model instance.
    """
    model = ultralytics.YOLO(checkpoint_path)
    model.to(device)
    logging.rank_zero_info(
        "Loaded YOLO model from %s on device %s",
        checkpoint_path,
        device,
    )
    return model


def _project_bbox_to_world(
    bbox: structure.BoundingBox,
    geo: structure.GeoReference,
) -> typing.Tuple[float, float]:
    r"""Projects the bbox bottom-center to world coordinates.

    The bottom-center is used because it lies on the road plane (modulo detection noise), unlike
    the bbox centroid which sits above it and would bias the projection.
    """
    u = (bbox.x1 + bbox.x2) / 2.0
    v = bbox.y2
    return georef_lib.pixel_to_world(geo, u, v)


def _open_video_or_raise(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    return cap


def extract_trajectories(
    model: ultralytics.YOLO,
    video_path: str,
    tracker_config: str = "botsort.yaml",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    vehicle_classes: typing.Optional[typing.FrozenSet[int]] = None,
    img_size: int = 1280,
    camera_params: typing.Optional[structure.CameraParameters] = None,
    georeference: typing.Optional[structure.GeoReference] = None,
) -> structure.TrajectorySet:
    r"""Runs detection + tracking on a video and returns a trajectory set.

    Args:
        model (ultralytics.YOLO): Loaded YOLO model.
        video_path (str): Path to the input video file (MP4 or AVI).
        tracker_config (str): Tracker configuration file name. Either
            ``"botsort.yaml"`` or ``"bytetrack.yaml"``.
        confidence_threshold (float): Minimum confidence to keep a detection.
        iou_threshold (float): IoU threshold for non-maximum suppression.
        vehicle_classes (Optional[FrozenSet[int]]): COCO class IDs to retain.
            Default is the ``{car, motorcycle, bus, truck}`` subset.
        img_size (int): Input image size for YOLO inference.
        camera_params (Optional[CameraParameters]): If provided, frames are
            undistorted before detection (frame-by-frame path).
        georeference (Optional[GeoReference]): If provided, each detection
            is projected to world coordinates and stored on the
            ``structure.Detection``.

    Returns:
        ``structure.TrajectorySet`` containing all tracked vehicle
        trajectories.
    """
    if vehicle_classes is None:
        vehicle_classes = _DEFAULT_VEHICLE_CLASSES

    cap = _open_video_or_raise(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    trajectory_set = structure.TrajectorySet(
        source_video=video_path,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        total_frames=total_frames,
    )

    if camera_params is None:
        results_iter = model.track(
            source=video_path,
            tracker=tracker_config,
            persist=True,
            stream=True,
            conf=confidence_threshold,
            iou=iou_threshold,
            classes=list(vehicle_classes),
            imgsz=img_size,
            verbose=False,
        )
        _consume_results(
            results_iter=results_iter,
            trajectory_set=trajectory_set,
            georeference=georeference,
            total_frames=total_frames,
        )
    else:
        _run_frame_by_frame(
            model=model,
            video_path=video_path,
            tracker_config=tracker_config,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            vehicle_classes=list(vehicle_classes),
            img_size=img_size,
            camera_params=camera_params,
            georeference=georeference,
            trajectory_set=trajectory_set,
            total_frames=total_frames,
        )

    logging.rank_zero_info(
        "Tracking complete: %d frames, %d unique tracks.",
        total_frames,
        len(trajectory_set.trajectories),
    )
    return trajectory_set


def _consume_results(
    results_iter,
    trajectory_set: structure.TrajectorySet,
    georeference: typing.Optional[structure.GeoReference],
    total_frames: int,
) -> None:
    r"""Iterates over ultralytics results and appends detections."""
    with tqdm_logging.logging_redirect_tqdm():
        pbar = tqdm.tqdm(
            total=total_frames,
            desc="Processing Frames",
            position=0,
            leave=False,
        )
        for frame_idx, result in enumerate(results_iter):
            pbar.update(1)
            _add_result_to_set(
                frame_idx=frame_idx,
                result=result,
                trajectory_set=trajectory_set,
                georeference=georeference,
            )
            pbar.set_postfix(
                {"Num Trajectories": len(trajectory_set.trajectories)},
                refresh=True,
            )
        pbar.close()


def _run_frame_by_frame(
    model: ultralytics.YOLO,
    video_path: str,
    tracker_config: str,
    confidence_threshold: float,
    iou_threshold: float,
    vehicle_classes: typing.List[int],
    img_size: int,
    camera_params: structure.CameraParameters,
    georeference: typing.Optional[structure.GeoReference],
    trajectory_set: structure.TrajectorySet,
    total_frames: int,
) -> None:
    r"""Reads, undistorts, and tracks frames one at a time."""
    cap = _open_video_or_raise(video_path)
    pbar = tqdm.tqdm(
        total=total_frames,
        desc="Processing (undistort)",
        position=0,
        leave=False,
    )
    frame_idx = 0
    with tqdm_logging.logging_redirect_tqdm():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = calibration_lib.undistort_frame(frame, camera_params)
                results = model.track(
                    source=frame,
                    tracker=tracker_config,
                    persist=True,
                    stream=False,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    classes=vehicle_classes,
                    imgsz=img_size,
                    verbose=False,
                )
                if results:
                    _add_result_to_set(
                        frame_idx=frame_idx,
                        result=results[0],
                        trajectory_set=trajectory_set,
                        georeference=georeference,
                    )
                frame_idx += 1
                pbar.update(1)
                pbar.set_postfix(
                    {"Num Trajectories": len(trajectory_set.trajectories)},
                    refresh=True,
                )
        finally:
            cap.release()
            pbar.close()


def _add_result_to_set(
    frame_idx: int,
    result,
    trajectory_set: structure.TrajectorySet,
    georeference: typing.Optional[structure.GeoReference],
) -> None:
    r"""Pulls `boxes/ids/cls/conf` out of one ultralytics Result and stores."""
    if result.boxes is None or result.boxes.id is None:
        return
    boxes = result.boxes
    ids = boxes.id
    assert isinstance(ids, torch.Tensor)

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(np.int64)
    confs = boxes.conf.cpu().numpy()
    track_ids = ids.cpu().numpy().astype(np.int64)
    names = result.names

    for i in range(len(boxes)):
        bbox = structure.BoundingBox(
            x1=float(xyxy[i, 0]),
            y1=float(xyxy[i, 1]),
            x2=float(xyxy[i, 2]),
            y2=float(xyxy[i, 3]),
        )
        world_x: typing.Optional[float] = None
        world_y: typing.Optional[float] = None
        if georeference is not None:
            world_x, world_y = _project_bbox_to_world(bbox, georeference)

        detection = structure.Detection(
            frame_index=frame_idx,
            track_id=int(track_ids[i]),
            bbox=bbox,
            class_id=int(cls_ids[i]),
            class_name=names[int(cls_ids[i])],
            confidence=float(confs[i]),
            world_x=world_x,
            world_y=world_y,
        )
        trajectory_set.add_detection(detection)
