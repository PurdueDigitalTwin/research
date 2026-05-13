r"""Helper functions for Ground-control-point (GCP) workflow.

Three operations make up the GCP preparation pipeline for INDOT drone footage:

1. ``detect_marked_gcps`` finds the hand-drawn white outlines in a
   ``*GCPs Marked.png`` image and reports per-marker centroid pixels.
2. ``assemble_gcp_json`` joins INDOT "simple" CSV world coordinates
   with a per-label ``label,u,v`` pixel CSV into the GCP JSON consumed by
   ``georeferencing`` module.
3. ``register_marked_to_video`` ORB+RANSAC registers a marked-photo
   to a video frame, transforming GCP pixel coords from the marked-photo
   frame to the video frame.
"""

import json
import os
import typing

import cv2
import numpy as np

from src.projects.jtrp.rci import serialization
from src.projects.jtrp.rci import structure
from src.utilities import logging


# --- Marker detection ---------------------------------------------------------
def detect_marked_gcps(
    image: np.ndarray,
    white_threshold: int = 235,
    min_area_px: int = 50,
    max_area_px: int = 20_000,
    max_markers: int = 8,
) -> typing.List[typing.Tuple[str, float, float, float]]:
    r"""Extracts candidate GCP pixel coordinates from a marked photo.

    The function thresholds for near-white pixels, closes small gaps in
    the hand-drawn outlines, finds external contours, filters by area,
    and returns the largest ``max_markers`` candidates sorted into a
    deterministic top-to-bottom / left-to-right order.

    Args:
        image (NDArray): BGR image (``cv2.imread`` result).
        white_threshold (int): Per-channel intensity threshold for the
            "near-white" mask. Default ``235``.
        min_area_px (int): Minimum contour area in ``\text{px}^2``.
            Default ``50``.
        max_area_px (int): Maximum contour area in ``\text{px}^2``.
            Reject the watermark / image borders by area cap. Default
            ``20\,000``.
        max_markers (int): Maximum candidates to return. Default ``8``.

    Returns:
        A list of ``(label, u, v, area_px)`` tuples where ``label`` is a
        1-based ``"1".."N"`` string in row-major reading order. The list
        is sorted in that reading order rather than by area.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR image; got shape {image.shape}."
        )

    mask = np.all(image >= white_threshold, axis=2).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    raw: typing.List[typing.Tuple[float, float, float]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area_px or area > max_area_px:
            continue
        moments = cv2.moments(cnt)
        if moments["m00"] == 0.0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        raw.append((area, float(cx), float(cy)))

    # Keep the largest by area first.
    raw.sort(key=lambda t: -t[0])
    raw = raw[:max_markers]
    # Then re-order by reading order.
    raw.sort(key=lambda t: (round(t[2] / 50), t[1]))

    return [
        (str(i), cx, cy, area) for i, (area, cx, cy) in enumerate(raw, start=1)
    ]


def save_marker_overlay(
    image: np.ndarray,
    markers: typing.Sequence[typing.Tuple[str, float, float, float]],
    output_path: str,
) -> None:
    r"""Saves a debug overlay with red circles and labels at each marker."""
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    overlay = image.copy()
    for label, u, v, _area in markers:
        cv2.circle(
            overlay,
            (int(round(u)), int(round(v))),
            radius=20,
            color=(0, 0, 255),
            thickness=3,
        )
        cv2.putText(
            overlay,
            label,
            (int(round(u)) + 24, int(round(v))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
    cv2.imwrite(output_path, overlay)


# --- GCP JSON assembly --------------------------------------------------------
def assemble_gcp_json(
    world_csv: str,
    image_uv_csv: typing.Optional[str],
    output_path: str,
    table_index: int = 1,
    world_units: str = "ftus",
    world_crs: str = "NAD83(2011) / Indiana East (ftUS)",
) -> typing.List[structure.GroundControlPoint]:
    r"""Builds a GCP JSON by joining a world-coord CSV with a pixel-coord CSV.

    If ``image_uv_csv`` is ``None``, GCPs are written with placeholder
    pixel coordinates ``(-1, -1)`` and a warning is logged. The downstream
    homography fit will refuse to use placeholder GCPs.

    Args:
        world_csv (str): INDOT "simple" CSV with world Northing/Easting.
        image_uv_csv (Optional[str]): CSV with ``label,u,v`` columns.
        output_path (str): Destination JSON path.
        table_index (int): 1-based table index inside the world CSV.
        world_units (str): Free-text world-units label for metadata.
        world_crs (str): Free-text world-CRS description for metadata.

    Returns:
        The list of ``GroundControlPoint`` instances written.
    """
    world_rows = serialization.load_world_csv(
        world_csv, table_index=table_index
    )
    uv_map = (
        serialization.load_image_uv_csv(image_uv_csv) if image_uv_csv else {}
    )

    gcps: typing.List[structure.GroundControlPoint] = []
    for label, world_x, world_y in world_rows:
        u, v = uv_map.get(label, (-1.0, -1.0))
        gcps.append(
            structure.GroundControlPoint(
                label=label,
                world_x=world_x,
                world_y=world_y,
                image_u=u,
                image_v=v,
            )
        )

    metadata: typing.Dict[str, typing.Any] = {
        "world_units": world_units,
        "world_crs": world_crs,
        "source_csv": os.path.abspath(world_csv),
        "table_index": int(table_index),
        "image_uv_csv": (
            os.path.abspath(image_uv_csv) if image_uv_csv else None
        ),
    }
    serialization.save_gcps_to_json(gcps, output_path, metadata=metadata)

    missing = [g.label for g in gcps if g.image_u < 0 or g.image_v < 0]
    if missing:
        logging.rank_zero_warning(
            "GCP JSON %s has %d placeholder pixel(s): %s",
            output_path,
            len(missing),
            ", ".join(missing),
        )
    else:
        logging.rank_zero_info("Wrote %d GCPs to %s", len(gcps), output_path)
    return gcps


# --- ORB-based marked-to-video registration ----------------------------------
def read_video_frame(video_path: str, time_s: float) -> np.ndarray:
    r"""Decodes a single frame from a video at the requested timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target = int(round(time_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise OSError(
            f"Could not read frame at t={time_s:.3f} s "
            f"(frame index {target})."
        )
    return frame


def _estimate_orb_homography(
    src: np.ndarray,
    dst: np.ndarray,
    n_features: int = 10_000,
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 4.0,
) -> typing.Tuple[np.ndarray, int, int]:
    r"""Estimates a src→dst homography via ORB + Lowe-ratio + RANSAC.

    Returns:
        ``(H, n_matches_after_ratio, n_inliers)``.
    """
    g_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)  # type: ignore[attr-defined]
    kp_src, des_src = orb.detectAndCompute(g_src, None)
    kp_dst, des_dst = orb.detectAndCompute(g_dst, None)
    if des_src is None or des_dst is None:
        raise RuntimeError("ORB failed to compute descriptors.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_src, des_dst, k=2)
    good: typing.List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
    if len(good) < 8:
        raise RuntimeError(
            f"Only {len(good)} good ORB matches found; need >= 8."
        )

    pts_src = np.array([kp_src[m.queryIdx].pt for m in good], dtype=np.float32)
    pts_src = pts_src.reshape(-1, 1, 2)
    pts_dst = np.array([kp_dst[m.trainIdx].pt for m in good], dtype=np.float32)
    pts_dst = pts_dst.reshape(-1, 1, 2)
    H, mask = cv2.findHomography(
        pts_src, pts_dst, cv2.RANSAC, ransac_threshold
    )
    if H is None:
        raise RuntimeError("RANSAC homography fit failed.")
    n_inliers = int(mask.sum()) if mask is not None else 0
    return H, len(good), n_inliers


def _warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    r"""Applies homography ``H`` to an array of points with shape ``(N, 2)``."""
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    proj = homog @ H.T
    return proj[:, :2] / proj[:, 2:3]


def _save_registration_overlay(
    marked: np.ndarray,
    video_frame: np.ndarray,
    marked_uv: np.ndarray,
    video_uv: np.ndarray,
    out_path: str,
) -> None:
    r"""Saves a debug overlay with marked GCPs and video frame side-by-side."""
    h1, w1 = marked.shape[:2]
    h2, w2 = video_frame.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:h1, :w1] = marked
    canvas[:h2, w1 : w1 + w2] = video_frame
    for (mu, mv), (vu, vv) in zip(marked_uv, video_uv):
        pa = (int(round(mu)), int(round(mv)))
        pb = (int(round(vu)) + w1, int(round(vv)))
        cv2.circle(canvas, pa, 18, (0, 0, 255), 3)
        cv2.circle(canvas, pb, 18, (0, 255, 0), 3)
        cv2.line(canvas, pa, pb, (255, 255, 0), 1)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    cv2.imwrite(out_path, canvas)


def register_marked_to_video(
    marked_image_path: str,
    video_path: str,
    video_time_s: float,
    gcp_json_in: str,
    gcp_json_out: str,
    orb_features: int = 10_000,
    ratio_threshold: float = 0.75,
    ransac_threshold_px: float = 4.0,
    debug_overlay_path: typing.Optional[str] = None,
) -> typing.Tuple[int, int]:
    r"""Warps GCP pixels from a marked-photo to a video frame.

    Reads the GCP JSON ``gcp_json_in``, computes an ORB+RANSAC homography
    between the marked photo and the video frame at ``video_time_s``,
    transforms each GCP's ``(image_u, image_v)``, and writes the updated
    GCP JSON to ``gcp_json_out``.

    Args:
        marked_image_path (str): Path to the image with GCP markers.
        video_path (str): Path to the video file.
        video_time_s (float): Timestamp in seconds for the video frame to
            register to.
        gcp_json_in (str): Path to the input GCP JSON with pixel coordinates.
        gcp_json_out (str): Path to write the output GCP JSON with updated
            pixel coordinates.
        orb_features (int): Number of ORB features to extract.
            Default is ``10000``.
        ratio_threshold (float): Threshold for Lowe's ratio test in ORB
            matching. Default is ``0.75``.
        ransac_threshold_px (float): Threshold in pixels for RANSAC inlier
            determination. Default is ``4.0``.
        debug_overlay_path (Optional[str]): Optional path to write a debug
            image showing the marked photo and video frame side-by-side
            with detected GCPs and matches overlaid. Default is ``None``.

    Returns:
        A tuple of ``(n_matches, n_inliers)`` reported by the RANSAC fit.
    """
    marked = cv2.imread(marked_image_path)
    if marked is None:
        raise OSError(f"Cannot read marked image: {marked_image_path}")
    video_frame = read_video_frame(video_path, video_time_s)

    H, n_matches, n_inliers = _estimate_orb_homography(
        marked,
        video_frame,
        n_features=orb_features,
        ratio_threshold=ratio_threshold,
        ransac_threshold=ransac_threshold_px,
    )
    logging.rank_zero_info(
        "ORB homography: %d good matches, %d RANSAC inliers.",
        n_matches,
        n_inliers,
    )

    with open(gcp_json_in) as f:
        gcp_data = json.load(f)
    marked_uv = np.array(
        [[g["image_u"], g["image_v"]] for g in gcp_data["gcps"]],
        dtype=np.float64,
    )
    video_uv = _warp_points(marked_uv, H)
    for i, g in enumerate(gcp_data["gcps"]):
        g["image_u"] = float(video_uv[i, 0])
        g["image_v"] = float(video_uv[i, 1])

    metadata = gcp_data.setdefault("metadata", {})
    metadata["registered_from"] = os.path.abspath(marked_image_path)
    metadata["registered_to_video"] = os.path.abspath(video_path)
    metadata["video_reference_time_s"] = float(video_time_s)
    metadata["registration_inliers"] = int(n_inliers)
    metadata["registration_matches"] = int(n_matches)

    out_dir = os.path.dirname(os.path.abspath(gcp_json_out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(gcp_json_out, "w") as f:
        json.dump(gcp_data, f, indent=2)
    logging.rank_zero_info(
        "Wrote %d registered GCPs to %s",
        len(gcp_data["gcps"]),
        gcp_json_out,
    )

    if debug_overlay_path is not None:
        _save_registration_overlay(
            marked,
            video_frame,
            marked_uv,
            video_uv,
            debug_overlay_path,
        )
        logging.rank_zero_info(
            "Wrote registration overlay to %s", debug_overlay_path
        )

    return n_matches, n_inliers
