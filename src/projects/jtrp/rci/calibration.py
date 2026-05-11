r"""Camera intrinsic calibration utilities with OpenCV.

This module provides functionality for deriving camera intrinsic parameters
from chessboard calibration images. See the official tutorial for more details:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html.

Example usage:

>>> import glob
>>> from src.projects.jtrp.rci import calibration
>>>
>>> # Load image files
>>> images = calibration.load_image_files(glob.glob("calibration_*.jpg"))
>>> images.extend(
>>>     calibration.sample_video_frames("Camera Clibration.mp4", fps=2.0))
>>> )
>>> pattern, _ = calibration.auto_detact_pattern_size(images)
>>> parasms = calibration.calibrate_from_img(images, pattern, square_size=0.025)
"""

import dataclasses
import typing

import cv2
from numpy import typing as npt
import numpy as np

# Constants
# NOTE: common chessboard inner-corner sizes in `(cols, rows)` format.
# The data is ordered by likelihood for typical calibration boards.
# Auto-detection will iterate through this list.
_COMMON_PATTERN_SIZE: typing.List[typing.Tuple[int, int]] = [
    (9, 6),
    (8, 6),
    (7, 6),
    (7, 7),
    (6, 6),
    (5, 5),
    (7, 5),
    (6, 5),
    (10, 7),
]


# Data structure
@dataclasses.dataclass
class CameraParameters:
    r"""Data container for camera intrinsics and distortion parameters.

    Attributes:
        camera_matrix (NDArray[float]): A three-by-three intrinsic matrix.
            :math:`K = [[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1]]`, where
            :math:`f_x, f_y` are the focal lengths in pixel units, :math:`s`
            is the skew (often zero), and :math:`c_x, c_y` are the principal
            point coordinates in pixel units.
        dist_coeffs (NDArray[float]): Distortion coefficients in OpenCV format.
            The number of coefficients depends on the distortion model used
            during calibration. For the common 5-parameter radial-tangential
            model, the order is :math:`[k_1, k_2, p_1, p_2, k_3]`, where
            :math:`k_i` are radial distortion coefficients and :math:`p_i` are
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


# Helper functions
def detect_chessboard_corners(
    img: cv2.typing.MatLike,
    pattern_size: typing.Tuple[int, int],
    use_sb: bool = True,
) -> typing.Optional[cv2.typing.MatLike]:
    r"""Detects inner corners of a chessboard pattern in the given image.

    Args:
        img (MatLike): Input image to detect corners of shape ``(H, W)``.
        pattern_size (Tuple[int, int]): The number of inner corners of the
            chessboard pattern in ``(cols, rows)`` format.
        use_sb (bool, optional): Whether to use ``findChessboardCornersSB``
            for detection. This method is more robust to distortion and partial
            views but may be slower. Defaults to ``True``.

    Returns:
        Refined corner array of with a shape of ``(N, 1, 2)`` in pixel units,
            where :math:`N` is the total number of detected corners. If no corners are detected, returns ``None``.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            img,
            pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok and corners is not None:
            return corners.astype(np.float32)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    ok, corners = cv2.findChessboardCorners(img, pattern_size, flags=flags)
    if not ok or corners is None:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(
        img,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return refined.astype(np.float32)
