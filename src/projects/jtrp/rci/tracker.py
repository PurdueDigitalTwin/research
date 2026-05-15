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

from absl import logging
import cv2
import numpy as np
import torch
from tqdm import auto as tqdm
from tqdm.contrib import logging as tqdm_logging
import ultralytics

from src.projects.jtrp.rci import calibration as calibration_lib
from src.projects.jtrp.rci import georeferencing as georef_lib
from src.projects.jtrp.rci import structure

# Vehicle class names retained from a detection result. Filtering by name
# rather than by integer ID lets the same code handle both COCO-trained
# detectors (yolo26l: car/truck/bus/motorcycle) and DOTA-trained OBB
# detectors (yolo26l-obb: "large vehicle"/"small vehicle"). Ultralytics is
# invoked with ``classes=None`` so all detections come through and we
# post-filter here.
_DEFAULT_VEHICLE_CLASS_NAMES: typing.FrozenSet[str] = frozenset(
    {
        # COCO class names.
        "car",
        "truck",
        "bus",
        "motorcycle",
        # class names for `yolo26-obb``.
        "large vehicle",
        "small vehicle",
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
    logging.info(
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


def _roi_to_contour(
    roi: typing.Optional[structure.RegionOfInterest],
) -> typing.Optional[np.ndarray]:
    r"""Materialises a ROI polygon as a ``cv2.pointPolygonTest``-ready contour.

    Returns ``None`` when ``roi`` is ``None`` so the caller can short-circuit
    the filter entirely.
    """
    if roi is None:
        return None
    return np.asarray(roi.polygon, dtype=np.float32).reshape(-1, 1, 2)


def _bbox_in_roi(
    bbox: structure.BoundingBox,
    roi_contour: typing.Optional[np.ndarray],
) -> bool:
    r"""Returns True if the bbox bottom-center falls inside (or on) the ROI.

    Uses the bottom-center because that is the canonical ground-contact proxy already used by the
    world-projection code; keeping the test consistent means a detection that gets a world coord
    also gets ROI- accepted (and vice versa).
    """
    if roi_contour is None:
        return True
    u = (bbox.x1 + bbox.x2) / 2.0
    v = bbox.y2
    return cv2.pointPolygonTest(roi_contour, (float(u), float(v)), False) >= 0


def extract_trajectories(
    model: ultralytics.YOLO,
    video_path: str,
    tracker_config: str = "botsort.yaml",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    vehicle_class_names: typing.Optional[typing.FrozenSet[str]] = None,
    img_size: int = 1280,
    camera_params: typing.Optional[structure.CameraParameters] = None,
    georeference: typing.Optional[structure.GeoReference] = None,
    roi: typing.Optional[structure.RegionOfInterest] = None,
) -> structure.TrajectorySet:
    r"""Runs detection + tracking on a video and returns a trajectory set.

    Args:
        model (ultralytics.YOLO): Loaded YOLO model.
        video_path (str): Path to the input video file (MP4 or AVI).
        tracker_config (str): Tracker configuration file name. Either
            ``"botsort.yaml"`` or ``"bytetrack.yaml"``.
        confidence_threshold (float): Minimum confidence to keep a detection.
        iou_threshold (float): IoU threshold for non-maximum suppression.
        vehicle_class_names (Optional[FrozenSet[str]]): If provided, only
            retain detections whose class name is within this set. If ``None``, retain all detections. Default is ``None``.
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
    if vehicle_class_names is None:
        vehicle_class_names = _DEFAULT_VEHICLE_CLASS_NAMES
    roi_contour = _roi_to_contour(roi)

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
            imgsz=img_size,
            verbose=False,
        )
        _consume_results(
            results_iter=results_iter,
            trajectory_set=trajectory_set,
            georeference=georeference,
            roi_contour=roi_contour,
            vehicle_class_names=vehicle_class_names,
            total_frames=total_frames,
        )
    else:
        _run_frame_by_frame(
            model=model,
            video_path=video_path,
            tracker_config=tracker_config,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            img_size=img_size,
            camera_params=camera_params,
            georeference=georeference,
            roi_contour=roi_contour,
            vehicle_class_names=vehicle_class_names,
            trajectory_set=trajectory_set,
            total_frames=total_frames,
        )

    logging.info(
        "Tracking complete: %d frames, %d unique tracks.",
        total_frames,
        len(trajectory_set.trajectories),
    )
    return trajectory_set


def _consume_results(
    results_iter,
    trajectory_set: structure.TrajectorySet,
    georeference: typing.Optional[structure.GeoReference],
    roi_contour: typing.Optional[np.ndarray],
    vehicle_class_names: typing.FrozenSet[str],
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
                roi_contour=roi_contour,
                vehicle_class_names=vehicle_class_names,
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
    img_size: int,
    camera_params: structure.CameraParameters,
    georeference: typing.Optional[structure.GeoReference],
    roi_contour: typing.Optional[np.ndarray],
    vehicle_class_names: typing.FrozenSet[str],
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
                    imgsz=img_size,
                    verbose=False,
                )
                if results:
                    _add_result_to_set(
                        frame_idx=frame_idx,
                        result=results[0],
                        trajectory_set=trajectory_set,
                        georeference=georeference,
                        roi_contour=roi_contour,
                        vehicle_class_names=vehicle_class_names,
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


def _extract_detection_tensors(
    result,
) -> typing.Optional[
    typing.Tuple[
        np.ndarray,
        typing.Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        typing.Mapping[int, str],
    ]
]:
    r"""Pulls per-detection arrays from an ultralytics Result.

    Prefers ``result.obb`` when the model is an OBB detector (yolo26-obb,
    yolo11-obb, etc.). Falls back to ``result.boxes`` for axis-aligned
    models. Returns ``None`` if the result has no tracked detections.

    Returns:
        ``(xyxy, obb_corners, cls_ids, confs, track_ids, names)`` where
        ``obb_corners`` is ``None`` for AABB models and an ``(N, 4, 2)``
        array of polygon corners for OBB models.
    """
    obb = getattr(result, "obb", None)
    if obb is not None and getattr(obb, "id", None) is not None:
        ids = obb.id
        assert isinstance(ids, torch.Tensor)
        xyxy = obb.xyxy.cpu().numpy()
        corners = obb.xyxyxyxy.cpu().numpy()
        cls_ids = obb.cls.cpu().numpy().astype(np.int64)
        confs = obb.conf.cpu().numpy()
        track_ids = ids.cpu().numpy().astype(np.int64)
        return xyxy, corners, cls_ids, confs, track_ids, result.names
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        ids = boxes.id
        assert isinstance(ids, torch.Tensor)
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(np.int64)
        confs = boxes.conf.cpu().numpy()
        track_ids = ids.cpu().numpy().astype(np.int64)
        return xyxy, None, cls_ids, confs, track_ids, result.names
    return None


def _ground_anchor(
    bbox: structure.BoundingBox,
    obb_corners: typing.Optional[np.ndarray],
) -> typing.Tuple[float, float]:
    r"""Returns the (u, v) ground-contact pixel for a detection.

    For OBB detections, the lowest-y corner of the rotated rectangle is closer to the actual ground
    footprint than the AABB's bottom edge. For AABB-only detections, we fall back to the bbox
    bottom-center.
    """
    if obb_corners is not None:
        idx = int(obb_corners[:, 1].argmax())
        return float(obb_corners[idx, 0]), float(obb_corners[idx, 1])
    return (bbox.x1 + bbox.x2) / 2.0, bbox.y2


def _add_result_to_set(
    frame_idx: int,
    result,
    trajectory_set: structure.TrajectorySet,
    georeference: typing.Optional[structure.GeoReference],
    roi_contour: typing.Optional[np.ndarray],
    vehicle_class_names: typing.FrozenSet[str],
) -> None:
    r"""Pulls per-detection data out of one ultralytics Result and stores.

    Handles both axis-aligned (``result.boxes``) and oriented
    (``result.obb``) detectors. The ground-contact anchor used for ROI
    filtering and world-coord projection is the bbox bottom-center for
    AABB detections and the lowest-y OBB corner for OBB detections.
    """
    extracted = _extract_detection_tensors(result)
    if extracted is None:
        return
    xyxy, obb_corners, cls_ids, confs, track_ids, names = extracted

    for i in range(len(xyxy)):
        class_name = names[int(cls_ids[i])]
        if class_name not in vehicle_class_names:
            continue

        bbox = structure.BoundingBox(
            x1=float(xyxy[i, 0]),
            y1=float(xyxy[i, 1]),
            x2=float(xyxy[i, 2]),
            y2=float(xyxy[i, 3]),
        )
        corners_i = obb_corners[i] if obb_corners is not None else None
        anchor_u, anchor_v = _ground_anchor(bbox, corners_i)

        if roi_contour is not None:
            inside = (
                cv2.pointPolygonTest(roi_contour, (anchor_u, anchor_v), False)
                >= 0
            )
            if not inside:
                continue

        world_x: typing.Optional[float] = None
        world_y: typing.Optional[float] = None
        if georeference is not None:
            world_x, world_y = georef_lib.pixel_to_world(
                georeference, anchor_u, anchor_v
            )

        detection = structure.Detection(
            frame_index=frame_idx,
            track_id=int(track_ids[i]),
            bbox=bbox,
            class_id=int(cls_ids[i]),
            class_name=class_name,
            confidence=float(confs[i]),
            world_x=world_x,
            world_y=world_y,
        )
        trajectory_set.add_detection(detection)
