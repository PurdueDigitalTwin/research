r"""Pixel-to-world georeferencing via a planar homography.

Given a set of ground control points (GCPs) with both image-pixel and
world-plane coordinates, this module estimates the
:math:`3 \times 3` homography :math:`H_{p \to w}` that maps a pixel
:math:`(u, v)` to a world-plane point :math:`(X, Y)`:

.. math::
    [X', Y', W']^{\top} = H_{p \to w}\, [u, v, 1]^{\top},
    \quad (X, Y) = (X' / W', Y' / W').

The planar assumption is reasonable for road-surface trajectory extraction
from a nearly-nadir drone view, provided that all GCPs lie on (or near)
the road plane and that vehicle bounding-box bottom-centers are projected.

A typical world frame is a local projected CRS such as
``NAD83(2011) / Indiana East (ftUS)``; the world units / CRS metadata is
preserved for downstream interoperability but is otherwise opaque to this
module.
"""

import dataclasses
import typing

import cv2
import numpy as np
from numpy import typing as npt


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

    Attributes:
        homography_pix_to_world (NDArray): :math:`3 \times 3` matrix that
            maps pixel coordinates to world-plane coordinates.
        homography_world_to_pix (NDArray): :math:`3 \times 3` matrix that
            maps world-plane coordinates back to pixel coordinates
            (cached inverse of ``homography_pix_to_world``).
        world_units (str): Free-text label of world units (e.g., ``"ftus"``,
            ``"m"``).
        world_crs (str): Free-text description of the world CRS, e.g.,
            ``"NAD83(2011) / Indiana East (ftUS)"``.
        rms_reprojection_error_px (float): Image-space round-trip error
            (pixels): the RMS distance between each input pixel GCP and the
            same GCP after going world→pixel through ``homography_world_to_pix``.
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


def compute_homography(
    gcps: typing.Sequence[GroundControlPoint],
    world_units: str = "ftus",
    world_crs: str = "NAD83(2011) / Indiana East (ftUS)",
    use_ransac: bool = False,
    ransac_threshold_px: float = 3.0,
) -> GeoReference:
    r"""Estimates a pixel→world homography from at least four GCPs.

    Args:
        gcps (Sequence[GroundControlPoint]): Ground control points.
            At least four are required; if more are provided, a
            least-squares fit (or RANSAC fit) is used.
        world_units (str, optional): Free-text label for world units.
            Default is ``"ftus"`` (US survey feet).
        world_crs (str, optional): Free-text description of the world CRS.
            Default is the Indiana East ftUS state plane.
        use_ransac (bool, optional): If `True`, fit with RANSAC instead of
            ordinary least-squares. Default is `False`.
        ransac_threshold_px (float, optional): RANSAC reprojection
            threshold in pixels. Default is :math:`3.0`.

    Returns:
        :class:`GeoReference` holding both homographies and the original GCPs.

    Raises:
        ValueError: If fewer than four GCPs are supplied.
        RuntimeError: If :func:`cv2.findHomography` returns no solution
            (degenerate configuration).
    """
    if len(gcps) < 4:
        raise ValueError(
            f"Need at least 4 GCPs for a homography fit; got {len(gcps)}."
        )

    image_pts = np.array(
        [[g.image_u, g.image_v] for g in gcps], dtype=np.float64
    )
    world_pts = np.array(
        [[g.world_x, g.world_y] for g in gcps], dtype=np.float64
    )

    method = cv2.RANSAC if use_ransac else 0
    H_pw, _ = cv2.findHomography(
        image_pts,
        world_pts,
        method=method,
        ransacReprojThreshold=float(ransac_threshold_px),
    )
    if H_pw is None:
        raise RuntimeError(
            "cv2.findHomography returned no solution; check GCP configuration."
        )
    H_pw = np.asarray(H_pw, dtype=np.float64)
    H_wp = np.linalg.inv(H_pw)

    # Image-space round-trip error.
    back = transform_points(world_pts, H_wp)
    rms_px = float(np.sqrt(np.mean(np.sum((back - image_pts) ** 2, axis=1))))

    return GeoReference(
        homography_pix_to_world=H_pw,
        homography_world_to_pix=H_wp,
        world_units=world_units,
        world_crs=world_crs,
        rms_reprojection_error_px=rms_px,
        gcps=list(gcps),
    )


def transform_points(
    points: npt.NDArray,
    homography: npt.NDArray,
) -> npt.NDArray:
    r"""Applies a :math:`3 \times 3` homography to an array of 2D points.

    Args:
        points (NDArray): Array of shape :math:`(N, 2)`.
        homography (NDArray): :math:`3 \times 3` projective transform.

    Returns:
        Array of shape :math:`(N, 2)` of transformed points.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"`points` must have shape (N, 2); got {points.shape}."
        )
    homog = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    proj = homog @ homography.T
    w = proj[:, 2:3]
    return proj[:, :2] / w


def pixel_to_world(
    geo: GeoReference,
    u: float,
    v: float,
) -> typing.Tuple[float, float]:
    r"""Maps a single pixel to world coordinates.

    Args:
        geo (GeoReference): The reference homography object.
        u (float): The pixel column to map.
        v (float): The pixel row to map.

    Returns:
        A tuple of ``(world_x, world_y)`` coordinates in the world CRS.
    """
    p = geo.homography_pix_to_world @ np.array([u, v, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


def world_to_pixel(
    geo: GeoReference,
    x: float,
    y: float,
) -> typing.Tuple[float, float]:
    r"""Maps a single world point to pixel coordinates.

    Args:
        geo (GeoReference): The reference homography object.
        x (float): The world-plane X coordinate to map.
        y (float): The world-plane Y coordinate to map.

    Returns:
        A tuple of ``(u, v)`` pixel coordinates in the image.
    """
    p = geo.homography_world_to_pix @ np.array([x, y, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])
