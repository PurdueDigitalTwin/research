import collections
import dataclasses
import typing

from numpy import typing as npt
import numpy as np


@dataclasses.dataclass
class BoundingBox:
    r"""Axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1 (float): Left edge x-coordinate.
        y1 (float): Top edge y-coordinate.
        x2 (float): Right edge x-coordinate.
        y2 (float): Bottom edge y-coordinate.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> typing.Tuple[float, float]:
        r"""Tuple[float, float]: 2D coordinates of the bounding box center."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        r"""float: Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        r"""float: height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        r"""float: Area of the rectangle bounding box."""
        return self.width * self.height


@dataclasses.dataclass
class Detection:
    r"""A single detection within one frame.

    Attributes:
        frame_index (int): Zero-based frame number.
        track_id (int): Assigned tracker ID (or -1 if untracked).
        bbox (int): Bounding box in pixel coordinates.
        class_id (int): Integer class ID from YOLO.
        class_name (str): Human-readable class label (e.g., "car", "truck").
        confidence (float): Detection confidence score in :math:`[0, 1]`.
    """

    frame_index: int
    track_id: int
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float


@dataclasses.dataclass
class Trajectory:
    r"""A trajectory consisting for a sequence of detection for an object.

    Attributes:
        track_id (int): Unique tracker-assigned ID.
        detections (List[Detection]): Ordered list of per-frame detections.
    """

    track_id: int
    detections: typing.List[Detection] = dataclasses.field(
        default_factory=list
    )

    @property
    def frame_indices(self) -> typing.List[int]:
        r"""List[int]: A list of frame indices the vehicle was detected."""
        return [d.frame_index for d in self.detections]

    @property
    def center_positions(self) -> npt.NDArray[np.float_]:
        r"""NDArray[float]: An array of bounding box center coordinates."""
        return np.array([d.bbox.center for d in self.detections])

    @property
    def bounding_boxes(self) -> npt.NDArray[np.float_]:
        r"""NDArray[float]: An array of `[x1, y1, x2, y2]` coordinates."""
        return np.array(
            [
                [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
                for d in self.detections
            ]
        )

    @property
    def confidence_scores(self) -> typing.List[float]:
        r"""List[float]: A list of detection confidence scores."""
        return [d.confidence for d in self.detections]

    @property
    def dominant_class(self) -> str:
        r"""str: The most frequently assigned class label."""
        counts = collections.Counter(d.class_name for d in self.detections)
        return counts.most_common(1)[0][0]


@dataclasses.dataclass
class TrajectorySet:
    r"""A Collection of all vehicle trajectories from a video.

    Attributes:
        source_video (str): Path to the source video file.
        frame_width (int): Video frame width in pixels.
        frame_height (int): Video frame height in pixels.
        fps (float): Video frames per second.
        total_frames (int): Total number of frames in the video.
        trajectories (Dict[str, Trajectory]): Mapping `track_id` to trajectory.
    """

    source_video: str
    frame_width: int
    frame_height: int
    fps: float
    total_frames: int
    trajectories: typing.Dict[int, Trajectory] = dataclasses.field(
        default_factory=dict,
    )

    def add_detection(self, detection: Detection) -> None:
        """Adds a detection to the corresponding trajectory.

        .. note::

            This method will automatically creates a new `Trajectory`
            if the `track_id` did not exist.

        Args:
            detection (Detection): The detection to add.
        """
        if not isinstance(detection, Detection):
            raise TypeError(
                "Expect `detection` to be an `Detection` instance, "
                f"but got {type(detection)} instead."
            )

        tid = detection.track_id
        if tid not in self.trajectories:
            self.trajectories[tid] = Trajectory(track_id=tid)
        self.trajectories[tid].detections.append(detection)
