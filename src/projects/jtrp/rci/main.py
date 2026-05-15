r"""Entry point for the JTRP-5023 RCI trajectory pipeline."""

import dataclasses
import glob
import os
import shutil
import sys
import typing

from absl import app
from absl import flags
from absl import logging
import cv2

from src.projects.jtrp.rci import calibration as calibration_lib
from src.projects.jtrp.rci import gcp_tools
from src.projects.jtrp.rci import serialization

# -- constants -----------------------------------------------------------------
# Resolved at import time so the welcome banner is picked up from the bazel
# runfiles tree alongside this module. The ``constants/`` directory is
# declared as ``data`` of the ``:main`` binary in the package BUILD.
_BANNER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "constants",
    "welcome.txt",
)

# ``BUILD_WORKING_DIRECTORY`` is set by ``bazelisk run`` to the shell's CWD,
# so the default lands wherever the user invoked the binary. When the program
# is launched outside bazel (e.g., from an IDE), we fall back to the process
# CWD.
_DEFAULT_WORK_DIR = os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd())

# YOLO detection checkpoints offered by the model selector. Local paths
# are resolved against ``BUILD_WORKING_DIRECTORY`` so they work whether
# the binary is launched from the bazel runfiles tree or directly. The
# empty-value last entry is a sentinel meaning "ask for a custom name /
# path" — selecting it falls back to the free-text prompt.
_YOLO_MODEL_OPTIONS: typing.Sequence[typing.Tuple[str, str]] = (
    (
        os.path.join(_DEFAULT_WORK_DIR, "data/checkpoints/yolo26l-obb.pt"),
        "yolo26 large OBB (local, oriented bbox — best for aerial views)",
    ),
    (
        os.path.join(_DEFAULT_WORK_DIR, "data/checkpoints/yolo26l.pt"),
        "yolo26 large (local, axis-aligned bbox)",
    ),
    ("yolo11n.pt", "yolo11 nano   (~5 MB, fastest stock)"),
    ("yolo11s.pt", "yolo11 small  (~22 MB)"),
    ("yolo11m.pt", "yolo11 medium (~50 MB)"),
    ("yolo11l.pt", "yolo11 large  (~85 MB)"),
    ("yolo11x.pt", "yolo11 xlarge (~110 MB)"),
    ("", "custom path or model name"),
)

# Keywords for users to skip optional steps.
_SKIP_KEYWORDS = ("skip", "none", "no", "-")

# Terminal size for rendering the welcome banner and step headers.
_TERMINAL_WIDTH = shutil.get_terminal_size((80, 20)).columns

# Total number of steps in the linear pipeline, used for rendering progress.
_TOTAL_STEPS = 5

# System specs
_OPENCV_VERSION = cv2.__version__
_PYTHON_VERSION = sys.version_info
_VERSION: str = "0.1.0"

# -- flags ---------------------------------------------------------------------
flags.DEFINE_string(
    name="work_dir",
    default=_DEFAULT_WORK_DIR,
    required=False,
    help=(
        "Base directory for pipeline outputs (calibration JSON, GCP JSON, "
        "trajectory runs). Defaults to $BUILD_WORKING_DIRECTORY (set by "
        "``bazelisk run``) or the current working directory."
    ),
)


@dataclasses.dataclass
class _SessionState:
    r"""Session-scoped state shared across pipeline steps."""

    work_dir: str


# ---------------------------------------------------------------------------
# Generic prompt helpers.
# ---------------------------------------------------------------------------
def _prompt(
    label: str,
    default: typing.Optional[str] = None,
    required: bool = True,
) -> str:
    r"""Prompts the user for a string with an optional default."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            value = input(f"{label}{suffix}: ").strip()
        except EOFError:
            value = ""
        if not value and default is not None:
            return default
        if value:
            return value
        if not required:
            return ""
        print("  (value required)")


def _prompt_int(label: str, default: typing.Optional[int] = None) -> int:
    r"""Prompts the user for an integer with an optional default."""
    raw = _prompt(label, str(default) if default is not None else None)
    try:
        return int(raw)
    except ValueError:
        print(f"  (not an int: {raw!r}); please try again")
        return _prompt_int(label, default)


def _prompt_float(label: str, default: typing.Optional[float] = None) -> float:
    r"""Prompts the user for a float with an optional default."""
    raw = _prompt(label, str(default) if default is not None else None)
    try:
        return float(raw)
    except ValueError:
        print(f"  (not a float: {raw!r}); please try again")
        return _prompt_float(label, default)


def _prompt_bool(label: str, default: bool = True) -> bool:
    r"""Prompts the user for a yes/no answer with a default."""
    raw = _prompt(label, "y" if default else "n")
    return raw.lower() in ("y", "yes", "true", "1")


def _prompt_path(
    label: str,
    must_exist: bool = True,
    default: typing.Optional[str] = None,
) -> str:
    r"""Prompts the user for a file path, optionally validating existence."""
    while True:
        path = os.path.expanduser(_prompt(label, default))
        if must_exist and not os.path.exists(path):
            print(f"  path does not exist: {path}; please try again")
            continue
        return path


def _prompt_optional_path(
    label: str,
    must_exist: bool = True,
) -> typing.Optional[str]:
    r"""Prompts the user for a file path, or blank to skip."""
    raw = _prompt(label + " (blank to skip)", default="", required=False)
    if not raw:
        return None
    path = os.path.expanduser(raw)
    if must_exist and not os.path.exists(path):
        print(f"  path does not exist: {path}; skipping")
        return None
    return path


def _prompt_choice(
    label: str,
    options: typing.Sequence[typing.Tuple[str, str]],
    default_index: int = 0,
) -> str:
    r"""Renders a numbered selector and returns the chosen value.

    Each entry in ``options`` is ``(value, description)``. The entry's
    ``value`` is what gets returned. An empty ``value`` is treated as a
    sentinel meaning "custom" / "other"; callers can detect it with
    ``if not return_value`` and fall back to a free-text prompt.
    """
    print(f"\n{label}:")
    width = max((len(v) for v, _ in options if v), default=0)
    for i, (value, desc) in enumerate(options, start=1):
        marker = " (default)" if i - 1 == default_index else ""
        if value:
            print(f"  [{i}] {value:<{width}}  {desc}{marker}")
        else:
            print(f"  [{i}] {desc}{marker}")
    while True:
        try:
            raw = input(f"  > [{default_index + 1}]: ").strip()
        except EOFError:
            raw = ""
        if not raw:
            return options[default_index][0]
        try:
            idx = int(raw) - 1
        except ValueError:
            print(f"    (not a number: {raw!r}); try again")
            continue
        if not (0 <= idx < len(options)):
            print(f"    out of range: {raw}; try again")
            continue
        return options[idx][0]


# ---------------------------------------------------------------------------
# Small wrappers that hide cv2 imports from menu functions.
# ---------------------------------------------------------------------------
def _load_images(paths: typing.Sequence[str]) -> typing.List:
    r"""Loads images from disk, validating that they can be read."""
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise OSError(f"Cannot read image: {p}")
        images.append(img)
    return images


def _sample_video_frames(
    video_path: str,
    fps: float,
    max_frames: int,
) -> typing.List:
    r"""Samples frames from a video at a specified rate."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(fps, 1e-3))))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------
def _load_welcome_banner(path: str = _BANNER_PATH) -> str:
    r"""Reads the welcome banner from ``constants/welcome.txt``.

    Returns an empty string if the file is missing or empty, so callers
    can fall back to ``_DEFAULT_BANNER``.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().rstrip("\n")
    except OSError:
        return ""


def _print_welcome(state: _SessionState) -> None:
    r"""Prints the welcome screen to console."""
    print("\033[H\033[J", end="")
    print("=" * _TERMINAL_WIDTH)
    banner = _load_welcome_banner()
    print(banner)
    print()
    print(f"JTRP SPR — 5023 Driver Behavior Data Pipeline {_VERSION}")
    print()
    print("System info:")
    print(f"Python {_PYTHON_VERSION.major}.{_PYTHON_VERSION.minor}")
    print(f"OpenCV {_OPENCV_VERSION}")
    print(f"Working directory: {state.work_dir}")
    print()
    print("Author: Juanwu Lu")
    print("2026 Purdue Digital Twin Lab, Purdue University.")
    print("=" * _TERMINAL_WIDTH)
    print()
    print("Linear pipeline: 5 sequential steps. At each step you can either")
    print("  - press Enter to run the step interactively, or")
    print("  - paste a path to an existing output file to skip the step, or")
    print(
        "  - type 'skip' to omit the step entirely (only for optional steps)."
    )
    print("Press Ctrl-C at any time to abort.")


# ---------------------------------------------------------------------------
# Pipeline step orchestration.
# ---------------------------------------------------------------------------
def _print_step_header(index: int, total: int, title: str) -> None:
    r"""Prints a formatted header for a pipeline step."""
    print()
    print("=" * 62)
    print(f"Step {index}/{total}: {title}")
    print("=" * 62)


def _prompt_step_input(
    file_label: str,
    skippable: bool,
) -> typing.Tuple[str, typing.Optional[str]]:
    r"""Asks the user how to fulfil a step.

    Args:
        file_label (str): A human-friendly label for the file the step produces
            (e.g., "calibration JSON" or "GCP CSV").
        skippable (bool): Whether the user should be given the option to skip
            the step entirely. This is only appropriate for steps that produce
            non-essential outputs (e.g., debug overlays) or that the user may
            want to run manually outside this script.

    Returns:
        ``("run", None)`` when the user wants to run the step interactively,
        ``("use", "/path/to/file")`` when they supplied an existing file,
        or ``("skip", None)`` when they typed a skip keyword (only allowed
        when ``skippable`` is ``True``).
    """
    options = [
        "Enter = run interactively",
        f"path = use existing {file_label}",
    ]
    if skippable:
        options.append("'skip' = omit this step")
    print("  (" + " | ".join(options) + ")")
    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            return "skip", None
        if raw == "":
            return "run", None
        if skippable and raw.lower() in _SKIP_KEYWORDS:
            return "skip", None
        path = os.path.expanduser(raw)
        if not os.path.isfile(path):
            print(f"    not a file: {path}; try again")
            continue
        print(f"    using existing: {path}")
        return "use", path


# ---------------------------------------------------------------------------
# Step bodies — each returns the path to the artifact it produced
# (or ``None`` if no artifact was produced).
# ---------------------------------------------------------------------------
def _step_calibrate(state: _SessionState) -> typing.Optional[str]:
    work_dir = state.work_dir
    img_glob = _prompt(
        "Image glob pattern (e.g., '.../Camera Calibration *.png'); "
        "blank to skip",
        default="",
        required=False,
    )
    video_path = _prompt_optional_path("Calibration video path")
    if not img_glob and not video_path:
        print("  no inputs provided; aborting this step")
        return None

    images: typing.List = []
    if img_glob:
        paths = sorted(glob.glob(os.path.expanduser(img_glob)))
        if not paths:
            print(f"  no files matched {img_glob}")
        else:
            print(f"  loading {len(paths)} image(s)")
            images.extend(_load_images(paths))
    if video_path:
        fps = _prompt_float("Sampling rate (Hz)", default=2.0)
        max_frames = _prompt_int("Max frames to sample", default=80)
        print(f"  sampling up to {max_frames} frame(s) at {fps:g} Hz")
        images.extend(_sample_video_frames(video_path, fps, max_frames))
    if not images:
        print("  no calibration views collected; aborting this step")
        return None

    if _prompt_bool("Auto-detect chessboard pattern?", default=True):
        print(f"  auto-detecting pattern across {len(images)} view(s) ...")
        pattern, hits = calibration_lib.auto_detect_pattern_size(images)
        print(
            f"  picked pattern {pattern} (detected in {hits} of "
            f"{len(images)} views)"
        )
    else:
        cols = _prompt_int("Chessboard inner-corner cols", default=9)
        rows = _prompt_int("Chessboard inner-corner rows", default=6)
        pattern = (cols, rows)

    square_size = _prompt_float(
        "Real-world side length per square (m)", default=0.025
    )
    output_path = os.path.expanduser(
        _prompt(
            "Output calibration JSON path",
            default=os.path.join(work_dir, "calibration.json"),
        )
    )

    params = calibration_lib.calibrate_from_images(
        images, pattern, square_size=square_size
    )
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    serialization.save_camera_parameters(params, output_path)
    logging.info(
        "Calibration RMS reprojection error: %.4f px (over %d views).",
        params.rms_reprojection_error,
        params.num_views,
    )
    logging.info("Wrote calibration to %s", output_path)
    return output_path


def _step_detect_markers(state: _SessionState) -> typing.Optional[str]:
    r"""Detects GCP marker candidates in a marked photo and export CSV."""

    work_dir = state.work_dir
    image_path = _prompt_path("Marked photo path (PNG)")
    output_csv = os.path.expanduser(
        _prompt(
            "Output image_uv CSV path",
            default=os.path.join(work_dir, "image_uv.csv"),
        )
    )
    max_markers = _prompt_int("Max markers to keep", default=8)
    overlay_path = _prompt(
        "Debug overlay PNG path (blank to skip)",
        default="",
        required=False,
    )

    image = cv2.imread(image_path)
    if image is None:
        print(f"  could not read image: {image_path}")
        return None
    markers = gcp_tools.detect_marked_gcps(image, max_markers=max_markers)
    if not markers:
        print("  no marker candidates detected; check thresholds")
        return None
    serialization.save_image_uv_csv(markers, output_csv)
    print(f"  wrote {len(markers)} candidate(s) to {output_csv}")
    for label, u, v, area in markers:
        print(f"    label={label}  u={u:.1f}  v={v:.1f}  area={area:.0f}")
    if overlay_path:
        gcp_tools.save_marker_overlay(
            image, markers, os.path.expanduser(overlay_path)
        )
        print(f"  wrote overlay to {overlay_path}")
    return output_csv


def _step_assemble_gcps(
    state: _SessionState,
    image_uv_csv: typing.Optional[str],
) -> typing.Optional[str]:
    work_dir = state.work_dir
    world_csv = _prompt_path("INDOT 'simple' GCP CSV path (Northing/Easting)")
    table_index = _prompt_int(
        "Table index inside the CSV (1-based)", default=1
    )
    if image_uv_csv is None:
        image_uv_csv = _prompt_optional_path("Image UV CSV path")
    else:
        print(f"  using image_uv CSV from previous step: {image_uv_csv}")
    output_json = os.path.expanduser(
        _prompt(
            "Output GCP JSON path",
            default=os.path.join(work_dir, "gcps.json"),
        )
    )
    world_units = _prompt("World units label", default="ftus")
    world_crs = _prompt(
        "World CRS description",
        default="NAD83(2011) / Indiana East (ftUS)",
    )
    gcp_tools.assemble_gcp_json(
        world_csv=world_csv,
        image_uv_csv=image_uv_csv,
        output_path=output_json,
        table_index=table_index,
        world_units=world_units,
        world_crs=world_crs,
    )
    return output_json


def _step_register_marked(
    state: _SessionState,
    gcp_json_in: typing.Optional[str],
) -> typing.Optional[str]:
    work_dir = state.work_dir
    marked_image = _prompt_path("Marked photo path (PNG)")
    video_path = _prompt_path("Video path")
    video_time = _prompt_float("Video reference time (s)", default=0.0)
    if gcp_json_in is None:
        gcp_json_in = _prompt_path("Input GCP JSON path (marked-photo pixels)")
    else:
        print(f"  using GCP JSON from previous step: {gcp_json_in}")
    gcp_json_out = os.path.expanduser(
        _prompt(
            "Output GCP JSON path (video-frame pixels)",
            default=os.path.join(work_dir, "gcps_registered.json"),
        )
    )
    overlay = _prompt(
        "Debug overlay PNG path (blank to skip)",
        default="",
        required=False,
    )
    n_matches, n_inliers = gcp_tools.register_marked_to_video(
        marked_image_path=marked_image,
        video_path=video_path,
        video_time_s=video_time,
        gcp_json_in=gcp_json_in,
        gcp_json_out=gcp_json_out,
        debug_overlay_path=os.path.expanduser(overlay) if overlay else None,
    )
    print(
        f"  matched {n_matches} ORB descriptors, "
        f"{n_inliers} RANSAC inliers"
    )
    return gcp_json_out


def _step_extract_trajectories(
    state: _SessionState,
    calibration_path: typing.Optional[str],
    gcp_json_path: typing.Optional[str],
) -> None:
    work_dir = state.work_dir
    video_path = _prompt_path("Video path")
    output_dir = os.path.expanduser(
        _prompt("Output directory", default=os.path.join(work_dir, "run"))
    )
    model_checkpoint = _prompt_choice(
        "YOLO model", _YOLO_MODEL_OPTIONS, default_index=0
    )
    if not model_checkpoint:
        model_checkpoint = _prompt(
            "YOLO checkpoint (path or name)", default="yolo11n.pt"
        )
    device = _prompt("Device (cpu / cuda:0 / mps)", default="cpu")
    tracker_config = _prompt(
        "Tracker (botsort.yaml / bytetrack.yaml)",
        default="botsort.yaml",
    )
    confidence = _prompt_float("Confidence threshold", default=0.25)
    iou = _prompt_float("IoU threshold", default=0.5)
    img_size = _prompt_int("YOLO input image size", default=1280)
    min_track_length = _prompt_int("Min track length to keep", default=5)
    output_format = _prompt("Output format (csv / json / both)", default="csv")
    render_video = _prompt_bool("Render annotated video?", default=True)
    if calibration_path:
        print(f"  using calibration from previous step: {calibration_path}")
    if gcp_json_path:
        print(f"  using GCPs from previous step: {gcp_json_path}")
    roi_path = _prompt_optional_path(
        "Region-of-interest JSON path (blank to disable filter)"
    )
    ngsim_output = _prompt(
        "NGSIM output CSV path (blank to skip)",
        default="",
        required=False,
    )
    ngsim_location = _prompt("NGSIM site name", default="JTRP-5023")

    # NOTE: torch loads here, after all prompts are answered, so a
    # CUDA-preload failure does not lose the user's inputs.
    from src.projects.jtrp.rci import pipeline

    config = pipeline.PipelineConfig(
        video_path=video_path,
        output_dir=output_dir,
        model_checkpoint=model_checkpoint,
        device=device,
        tracker_config=tracker_config,
        confidence_threshold=confidence,
        iou_threshold=iou,
        img_size=img_size,
        min_track_length=min_track_length,
        output_format=output_format,
        render_video=render_video,
        calibration_path=calibration_path,
        gcp_json_path=gcp_json_path,
        georeference_path=None,
        roi_path=roi_path,
        ngsim_output=(
            os.path.expanduser(ngsim_output) if ngsim_output else None
        ),
        ngsim_location=ngsim_location,
    )
    artifacts = pipeline.run_pipeline(config)
    print(
        f"  produced {artifacts.num_trajectories_kept} of "
        f"{artifacts.num_trajectories_total} trajectories"
    )
    for label, path in (
        ("trajectories CSV", artifacts.trajectories_csv),
        ("trajectories JSON", artifacts.trajectories_json),
        ("NGSIM CSV", artifacts.ngsim_csv),
        ("annotated video", artifacts.annotated_video),
    ):
        if path:
            print(f"    {label}: {path}")


# ---------------------------------------------------------------------------
# Top-level linear pipeline driver.
# ---------------------------------------------------------------------------
def _run_step(
    index: int,
    title: str,
    file_label: str,
    runner: typing.Callable[[], typing.Optional[str]],
    skippable: bool,
) -> typing.Optional[str]:
    r"""Drives one pipeline step.

    Prints the header, asks the user how to fulfil the step, and either
    invokes ``runner()`` (interactive run) or returns the existing path
    the user supplied. Returns ``None`` when the step is skipped.
    """
    _print_step_header(index, _TOTAL_STEPS, title)
    decision, path = _prompt_step_input(file_label, skippable=skippable)
    if decision == "use":
        return path
    if decision == "skip":
        print("  (step skipped)")
        return None
    try:
        return runner()
    except Exception as exc:  # noqa: BLE001
        logging.exception("Step %d failed: %s", index, exc)
        return None


def main(argv: typing.List[str]) -> int:
    del argv  # unused
    work_dir = os.path.expanduser(flags.FLAGS.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    logging.info("Working directory %s.", work_dir)
    state = _SessionState(work_dir=work_dir)
    _print_welcome(state)
    try:
        calibration_path = _run_step(
            index=1,
            title="Camera calibration (chessboard -> intrinsics JSON)",
            file_label="calibration JSON",
            runner=lambda: _step_calibrate(state),
            skippable=True,
        )

        image_uv_csv = _run_step(
            index=2,
            title="GCP marker detection (from 'Marked' photo)",
            file_label="image_uv CSV",
            runner=lambda: _step_detect_markers(state),
            skippable=True,
        )

        gcps_unregistered = _run_step(
            index=3,
            title="Assemble GCP JSON (world CSV + image_uv CSV)",
            file_label="GCP JSON",
            runner=lambda: _step_assemble_gcps(state, image_uv_csv),
            skippable=True,
        )

        gcps_registered = _run_step(
            index=4,
            title="Register marked photo to a video frame",
            file_label="registered GCP JSON",
            runner=lambda: _step_register_marked(state, gcps_unregistered),
            skippable=True,
        )

        # Trajectory extraction prefers registered GCPs; falls back to the
        # unregistered set if step 4 was skipped.
        final_gcps = gcps_registered or gcps_unregistered

        _print_step_header(
            _TOTAL_STEPS,
            _TOTAL_STEPS,
            "Trajectory extraction (YOLO + tracker + NGSIM)",
        )
        print("  (final step: runs YOLO detection, tracking, and export)")
        try:
            _step_extract_trajectories(
                state,
                calibration_path=calibration_path,
                gcp_json_path=final_gcps,
            )
        except KeyboardInterrupt:
            print("\n  (interrupted)")
            return 1
        except Exception as exc:  # noqa: BLE001
            logging.exception("Trajectory extraction failed: %s", exc)
            return 1

        print()
        print("Pipeline complete! Review the outputs above for next steps.")
        return 0
    except KeyboardInterrupt:
        print()
        print("Aborted.")
        return 1


if __name__ == "__main__":
    app.run(main=main)
