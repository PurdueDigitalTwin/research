r"""End-to-end trajectory extraction pipeline."""

import dataclasses
import os
import typing

from src.projects.jtrp.rci import calibration as calibration_lib
from src.projects.jtrp.rci import georeferencing as georef_lib
from src.projects.jtrp.rci import ngsim as ngsim_lib
from src.projects.jtrp.rci import serialization
from src.projects.jtrp.rci import structure
from src.projects.jtrp.rci import tracker
from src.projects.jtrp.rci import visualization
from src.utilities import logging


@dataclasses.dataclass
class PipelineConfig:
    r"""Inputs to ``run_pipeline``.

    Attributes:
        video_path (str): Source video path.
        output_dir (str): Destination directory for all artifacts.
        model_checkpoint (str): Path to a YOLO ``.pt`` file or model name.
        device (str): Inference device (``"cpu"``, ``"cuda:0"``, ``"mps"``).
        tracker_config (str): ``"botsort.yaml"`` or ``"bytetrack.yaml"``.
        confidence_threshold (float): Min detection confidence.
        iou_threshold (float): NMS IoU threshold.
        img_size (int): YOLO input image size.
        min_track_length (int): Drop trajectories with fewer detections
            than this.
        output_format (str): ``"csv"``, ``"json"``, or ``"both"`` for
            the raw trajectory dump.
        render_video (bool): If ``True``, render an annotated MP4.
        trail_length (int): Frame trail length for the annotated video.
        calibration_path (Optional[str]): Path to a calibration JSON.
            If supplied, frames are undistorted before detection.
        gcp_json_path (Optional[str]): Path to a GCP JSON used to fit
            the homography.
        georeference_path (Optional[str]): Path to a pre-fit GeoReference
            JSON. Takes precedence over ``gcp_json_path``.
        ngsim_output (Optional[str]): If non-empty, also export NGSIM
            CSV here. Requires georeference.
        ngsim_location (str): Site name written to the NGSIM ``Location``
            column.
        ngsim_smoothing_window (int): Moving-average half-window (frames)
            for the NGSIM velocity/acceleration computation.
    """

    video_path: str
    output_dir: str
    model_checkpoint: str = "yolo11n.pt"
    device: str = "cpu"
    tracker_config: str = "botsort.yaml"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    img_size: int = 1280
    min_track_length: int = 5
    output_format: str = "csv"
    render_video: bool = True
    trail_length: int = 30
    calibration_path: typing.Optional[str] = None
    gcp_json_path: typing.Optional[str] = None
    georeference_path: typing.Optional[str] = None
    ngsim_output: typing.Optional[str] = None
    ngsim_location: str = "JTRP-5023"
    ngsim_smoothing_window: int = 2


@dataclasses.dataclass
class PipelineArtifacts:
    r"""Paths to artifacts produced by ``run_pipeline``."""

    trajectories_csv: typing.Optional[str] = None
    trajectories_json: typing.Optional[str] = None
    ngsim_csv: typing.Optional[str] = None
    annotated_video: typing.Optional[str] = None
    num_trajectories_total: int = 0
    num_trajectories_kept: int = 0


def _maybe_load_calibration(
    path: typing.Optional[str],
) -> typing.Optional[structure.CameraParameters]:
    r"""Optionally loads camera calibration parameters from a JSON file."""
    if path is None:
        return None
    logging.rank_zero_info("Loading calibration parameters from %s", path)
    return serialization.load_camera_parameters(path)


def _maybe_build_georeference(
    gcp_json_path: typing.Optional[str],
    georef_path: typing.Optional[str],
) -> typing.Optional[structure.GeoReference]:
    if georef_path is not None:
        logging.rank_zero_info(
            "Loading pre-computed GeoReference from %s", georef_path
        )
        return serialization.load_georeference(georef_path)
    if gcp_json_path is None:
        return None
    logging.rank_zero_info(
        "Computing GeoReference from GCPs at %s", gcp_json_path
    )
    gcps = serialization.load_gcps_from_json(gcp_json_path)
    bad = [g.label for g in gcps if g.image_u < 0 or g.image_v < 0]
    if bad:
        raise ValueError(
            f"GCP JSON {gcp_json_path} contains placeholder pixels for "
            f"label(s): {bad}. Edit the file before running."
        )
    geo = georef_lib.compute_homography(gcps)
    logging.rank_zero_info(
        "GeoReference RMS reprojection error: %.3f px (over %d GCPs).",
        geo.rms_reprojection_error_px,
        len(gcps),
    )
    return geo


def run_pipeline(config: PipelineConfig) -> PipelineArtifacts:
    r"""Runs the full trajectory pipeline.

    Steps:
      1. Load the YOLO model.
      2. Load optional camera calibration and GeoReference.
      3. Run detection + tracking (undistorting frames if calibration is
         supplied, projecting detections to world coords if georeference
         is supplied).
      4. Filter short trajectories.
      5. Serialize trajectories (CSV/JSON), optionally as NGSIM, and
         render an annotated video.

    Args:
        config (PipelineConfig): All input configurations.

    Returns:
        A ``PipelineArtifacts`` instance with paths to every file written.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    logging.rank_zero_info("Loading YOLO model: %s", config.model_checkpoint)
    model = tracker.load_model(
        checkpoint_path=config.model_checkpoint,
        device=config.device,
    )
    model_name = os.path.splitext(os.path.basename(config.model_checkpoint))[0]

    camera_params = _maybe_load_calibration(config.calibration_path)
    georeference = _maybe_build_georeference(
        config.gcp_json_path, config.georeference_path
    )

    logging.rank_zero_info("Processing video: %s", config.video_path)
    trajectory_set = tracker.extract_trajectories(
        model=model,
        video_path=config.video_path,
        tracker_config=config.tracker_config,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        img_size=config.img_size,
        camera_params=camera_params,
        georeference=georeference,
    )

    total = len(trajectory_set.trajectories)
    if config.min_track_length > 0:
        filtered = {
            tid: traj
            for tid, traj in trajectory_set.trajectories.items()
            if len(traj.detections) >= config.min_track_length
        }
        removed = total - len(filtered)
        trajectory_set.trajectories = filtered
        logging.rank_zero_info(
            "Filtered %d short trajectories (min_length=%d); %d remain.",
            removed,
            config.min_track_length,
            len(filtered),
        )
    kept = len(trajectory_set.trajectories)

    video_basename = os.path.splitext(os.path.basename(config.video_path))[0]
    artifacts = PipelineArtifacts(
        num_trajectories_total=total,
        num_trajectories_kept=kept,
    )

    if config.output_format in ("csv", "both"):
        csv_path = os.path.join(
            config.output_dir,
            f"{model_name}_{video_basename}_trajectories.csv",
        )
        serialization.save_trajectories_csv(trajectory_set, csv_path)
        artifacts.trajectories_csv = csv_path
    if config.output_format in ("json", "both"):
        json_path = os.path.join(
            config.output_dir,
            f"{model_name}_{video_basename}_trajectories.json",
        )
        serialization.save_trajectories_json(trajectory_set, json_path)
        artifacts.trajectories_json = json_path
    if config.output_format not in ("csv", "json", "both"):
        logging.rank_zero_warning(
            "Unsupported output_format %r; skipping trajectory dump.",
            config.output_format,
        )

    if config.ngsim_output:
        if georeference is None:
            logging.rank_zero_warning(
                "Skipping NGSIM export: no GeoReference is configured."
            )
        else:
            ngsim_cfg = ngsim_lib.NgsimExportConfig(
                location=config.ngsim_location,
                smoothing_window=config.ngsim_smoothing_window,
                min_track_length=config.min_track_length,
            )
            ngsim_lib.save_ngsim_csv(
                trajectory_set, config.ngsim_output, ngsim_cfg
            )
            artifacts.ngsim_csv = config.ngsim_output

    if config.render_video:
        annotated_path = os.path.join(
            config.output_dir,
            f"{model_name}_{video_basename}_annotated.mp4",
        )
        visualization.render_annotated_video(
            trajectory_set=trajectory_set,
            output_path=annotated_path,
            trail_length=config.trail_length,
        )
        artifacts.annotated_video = annotated_path

    logging.rank_zero_info(
        "Pipeline complete. Outputs in: %s", config.output_dir
    )
    return artifacts
