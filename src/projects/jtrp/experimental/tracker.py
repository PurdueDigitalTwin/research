import typing

import cv2
import torch
from tqdm import auto as tqdm
from tqdm.contrib import logging as tqdm_logging
import ultralytics

from src.projects.jtrp.experimental import structure
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
        checkpoint_path (str): Path to .pt weights file or model name
            (e.g., "yolo11n.pt").
        device (str, optional): Device string (`"cpu"`, `"cuda:0"`, `"mps"`).
            Default is `"cpu"`.

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


def extract_trajectories(
    model: ultralytics.YOLO,
    video_path: str,
    tracker_config: str = "botsort.yaml",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    vehicle_classes: typing.Optional[typing.FrozenSet[int]] = None,
    img_size: int = 1280,
) -> structure.TrajectorySet:
    r"""Runs detection and tracking on a video and returns trajectories.

    .. note::

        This function calls the `ultralytics` streaming tracker API to process
        frames one at a time, keeping memory usage constant regardless of
        the length of the video stream.

    Args:
        model (ultralytics.YOLO): Loaded YOLO model.
        video_path (str): Path to the input video file (MP4 or AVI).
        tracker_config (str): Tracker configuration file name. Either
            "botsort.yaml" or "bytetrack.yaml". Default is `"botsort.yaml"`.
        confidence_threshold (float, optional): Minimum confidence to keep
            the detection. Default is :math:`0.25`.
        iou_threshold (float, optional): IoU threshold for non-maximum
            suppression. Default is :math:`0.5`.
        vehicle_classes (Optional[FrozenSet[int]]): Set of COCO class IDs to
            retain. If `None`, defaults to the class labels of `cars`,
            `motorcycles`, `buses`, and `trucks`.
        img_size (int, optional): Input image size for YOLO inference.
            Default is :math:`1280`.

    Returns:
        A TrajectorySet containing all tracked vehicle trajectories.
    """
    if vehicle_classes is None:
        vehicle_classes = _DEFAULT_VEHICLE_CLASSES

    # Read video metadata via OpenCV.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
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

    # Run tracking with ultralytics stream mode.
    results = model.track(
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

    with tqdm_logging.logging_redirect_tqdm():
        pbar = tqdm.tqdm(
            total=total_frames,
            desc="Processing Frames",
            position=0,
            leave=False,
        )
        for frame_idx, result in enumerate(results):
            pbar.update(1)
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            ids = boxes.id
            assert isinstance(ids, torch.Tensor)

            for i in range(len(boxes)):
                track_id = int(ids[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                class_name = result.names[class_id]

                detection = structure.Detection(
                    frame_index=frame_idx,
                    track_id=track_id,
                    bbox=structure.BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )
                trajectory_set.add_detection(detection)

            pbar.set_postfix(
                {"Num Trajectories": len(trajectory_set.trajectories)},
                refresh=True,
            )

    pbar.close()
    logging.rank_zero_info(
        "Tracking complete: %d frames, %d unique tracks.",
        total_frames,
        len(trajectory_set.trajectories),
    )
    return trajectory_set
