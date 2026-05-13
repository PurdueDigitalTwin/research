import collections
import dataclasses
import typing

from numpy import typing as npt
import numpy as np


@dataclasses.dataclass
class CameraParameters:
    r"""Data container for camera intrinsics and distortion parameters.

    Attributes:
        camera_matrix (NDArray[float]): A three-by-three intrinsic matrix.
            ``K = [[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1]]``, where
            ``f_x, f_y`` are the focal lengths in pixel units, ``s``
            is the skew (often zero), and ``c_x, c_y`` are the principal
            point coordinates in pixel units.
        dist_coeffs (NDArray[float]): Distortion coefficients in OpenCV format.
            The number of coefficients depends on the distortion model used
            during calibration. For the common 5-parameter radial-tangential
            model, the order is ``[k_1, k_2, p_1, p_2, k_3]``, where
            ``k_i`` are radial distortion coefficients and ``p_i`` are
            tangential distortion coefficients.
        img_size (Tuple[int, int]): The width and height of the image used
            for calibration in pixels (px).
        rms_reprojection_error (float): Projection error in pixel units (px).
            This is the root mean square (RMS) of the reprojection error across
            all calibration images and detected corners. It quantifies how well
            the estimated intrinsics explain the observed corner positions.
        pattern_size (Tuple[int, int]): The number of inner corners of the
            calibration chessboard pattern in ``(cols, rows)`` format.
        square_size (float): The side length per square of the calibration
            chessboard pattern in meters (m).
        num_views (int): The number of calibration images (views) that
            contribute to the corner detections.
    """

    camera_matrix: npt.NDArray[np.float64]
    dist_coeffs: npt.NDArray[np.float64]
    img_size: typing.Tuple[int, int]
    rms_reprojection_error: float
    pattern_size: typing.Tuple[int, int]
    square_size: float
    num_views: int

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        r"""Returns a serializable dictionary of the camera parameters."""
        return dict(
            camera_matrix=self.camera_matrix.tolist(),
            dist_coeffs=self.dist_coeffs.tolist(),
            img_size=list(int(s) for s in self.img_size),
            rms_reprojection_error=float(self.rms_reprojection_error),
            pattern_size=list(int(s) for s in self.pattern_size),
            square_size=float(self.square_size),
            num_views=int(self.num_views),
        )

    @classmethod
    def from_dict(
        cls: typing.Type["CameraParameters"],
        data: typing.Dict[str, typing.Any],
    ) -> "CameraParameters":
        r"""Constructs a ``CameraParameters`` instance from a dictionary."""
        img_size = [int(s) for s in data["img_size"]]
        pattern_size = [int(s) for s in data["pattern_size"]]
        return cls(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
            img_size=(img_size[0], img_size[1]),
            rms_reprojection_error=float(data["rms_reprojection_error"]),
            pattern_size=(pattern_size[0], pattern_size[1]),
            square_size=float(data["square_size"]),
            num_views=int(data["num_views"]),
        )


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
        bbox (BoundingBox): Bounding box in pixel coordinates.
        class_id (int): Integer class ID from YOLO.
        class_name (str): Human-readable class label (e.g., "car", "truck").
        confidence (float): Detection confidence score in ``[0, 1]``.
        world_x (Optional[float]): World-plane X (e.g., Easting in ftUS) of
            the projected bounding-box bottom-center. ``None`` if no
            georeferencing was applied.
        world_y (Optional[float]): World-plane Y (e.g., Northing in ftUS)
            of the projected bounding-box bottom-center. ``None`` if no
            georeferencing was applied.
    """

    frame_index: int
    track_id: int
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    world_x: typing.Optional[float] = None
    world_y: typing.Optional[float] = None


@dataclasses.dataclass
class Trajectory:
    r"""A trajectory consisting of a sequence of detection for an object.

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
    def center_positions(self) -> npt.NDArray[np.float64]:
        r"""NDArray[float]: An array of bounding box center coordinates."""
        return np.array([d.bbox.center for d in self.detections])

    @property
    def bounding_boxes(self) -> npt.NDArray[np.float64]:
        r"""NDArray[float]: An array of ``[x1, y1, x2, y2]`` coordinates."""
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
    def world_positions(self) -> typing.Optional[npt.NDArray[np.float64]]:
        r"""Optional[NDArray[float]]: ``(N, 2)`` world-plane coordinates.

        Returns ``None`` if any detection lacks a world coordinate.
        """
        if not self.detections:
            return None
        coords = []
        for d in self.detections:
            if d.world_x is None or d.world_y is None:
                return None
            coords.append((d.world_x, d.world_y))
        return np.array(coords, dtype=np.float64)

    @property
    def dominant_class(self) -> typing.Optional[str]:
        r"""Optional[str]: The most frequently assigned class label."""
        if not self.detections:
            return None

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
        trajectories (Dict[int, Trajectory]): Mapping ``track_id`` to trajectory.
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

            This method will automatically creates a new ``Trajectory``
            if the ``track_id`` did not exist.

        Args:
            detection (Detection): The detection to add.
        """
        if not isinstance(detection, Detection):
            raise TypeError(
                "Expect ``detection`` to be a ``Detection`` instance, "
                f"but got {type(detection)} instead."
            )

        tid = detection.track_id
        if tid not in self.trajectories:
            self.trajectories[tid] = Trajectory(track_id=tid)
        self.trajectories[tid].detections.append(detection)


@dataclasses.dataclass
class GroundControlPoint:
    r"""A single ground control point.

    Attributes:
        label (str): Human-readable identifier (e.g., ``"1"``, ``"GCP-A"``).
        world_x (float): World-plane X (e.g., Easting in ftus).
        world_y (float): World-plane Y (e.g., Northing in ftus).
        image_u (float): Image pixel column.
        image_v (float): Image pixel row.
    """

    label: str
    world_x: float
    world_y: float
    image_u: float
    image_v: float

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            "label": self.label,
            "world_x": float(self.world_x),
            "world_y": float(self.world_y),
            "image_u": float(self.image_u),
            "image_v": float(self.image_v),
        }

    @classmethod
    def from_dict(
        cls,
        data: typing.Dict[str, typing.Any],
    ) -> "GroundControlPoint":
        return cls(
            label=str(data["label"]),
            world_x=float(data["world_x"]),
            world_y=float(data["world_y"]),
            image_u=float(data["image_u"]),
            image_v=float(data["image_v"]),
        )


@dataclasses.dataclass
class GeoReference:
    r"""A pixel↔world planar homography with provenance metadata.

    The on-the-wire shape that backs georeferencing: dataclass fields here,
    constructor (``georeferencing.compute_homography``) and mapping
    helpers (``georeferencing.pixel_to_world``, etc.) co-located with
    the planar-projection logic.

    Attributes:
        homography_pix_to_world (NDArray): ``3 \times 3`` matrix that
            maps pixel coordinates to world-plane coordinates.
        homography_world_to_pix (NDArray): ``3 \times 3`` matrix that
            maps world-plane coordinates back to pixel coordinates
            (cached inverse of ``homography_pix_to_world``).
        world_units (str): Free-text label of world units (e.g., ``"ftus"``,
            ``"m"``).
        world_crs (str): Free-text description of the world CRS, e.g.,
            ``"NAD83(2011) / Indiana East (ftUS)"``.
        rms_reprojection_error_px (float): Image-space round-trip error
            (pixels): the RMS distance between each input pixel GCP and the
            same GCP after going world→pixel through
            ``homography_world_to_pix``.
        gcps (List[GroundControlPoint]): The GCPs used to fit the homography.
    """

    homography_pix_to_world: npt.NDArray
    homography_world_to_pix: npt.NDArray
    world_units: str
    world_crs: str
    rms_reprojection_error_px: float
    gcps: typing.List[GroundControlPoint] = dataclasses.field(
        default_factory=list
    )

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            "homography_pix_to_world": self.homography_pix_to_world.tolist(),
            "homography_world_to_pix": self.homography_world_to_pix.tolist(),
            "world_units": self.world_units,
            "world_crs": self.world_crs,
            "rms_reprojection_error_px": float(self.rms_reprojection_error_px),
            "gcps": [g.to_dict() for g in self.gcps],
        }

    @classmethod
    def from_dict(
        cls,
        data: typing.Dict[str, typing.Any],
    ) -> "GeoReference":
        return cls(
            homography_pix_to_world=np.asarray(
                data["homography_pix_to_world"], dtype=np.float64
            ),
            homography_world_to_pix=np.asarray(
                data["homography_world_to_pix"], dtype=np.float64
            ),
            world_units=str(data["world_units"]),
            world_crs=str(data["world_crs"]),
            rms_reprojection_error_px=float(data["rms_reprojection_error_px"]),
            gcps=[
                GroundControlPoint.from_dict(g) for g in data.get("gcps", [])
            ],
        )
