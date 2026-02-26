import os
import typing

from absl import app
from absl import flags

from src.projects.jtrp.experimental import serialization
from src.projects.jtrp.experimental import tracker
from src.projects.jtrp.experimental import visualization
from src.utilities import logging

flags.DEFINE_string(
    name="video_path",
    default=None,
    required=True,
    help="Path to input video file (MP4 or AVI).",
)
flags.DEFINE_string(
    name="model_checkpoint",
    default="yolo11n.pt",
    required=False,
    help="Path to YOLO model weights or model name.",
)
flags.DEFINE_string(
    name="output_dir",
    default=None,
    required=True,
    help="Directory for output files (trajectories and annotated video).",
)
flags.DEFINE_string(
    name="device",
    default="cpu",
    required=False,
    help="Device for inference: 'cpu', 'cuda:0', 'mps'.",
)
flags.DEFINE_string(
    name="tracker_config",
    default="botsort.yaml",
    required=False,
    help="Tracker configuration: 'botsort.yaml' or 'bytetrack.yaml'.",
)
flags.DEFINE_float(
    name="confidence_threshold",
    default=0.25,
    required=False,
    help="Minimum detection confidence threshold.",
)
flags.DEFINE_float(
    name="iou_threshold",
    default=0.5,
    required=False,
    help="IoU threshold for NMS during detection.",
)
flags.DEFINE_integer(
    name="img_size",
    default=1280,
    required=False,
    help="Input image size for YOLO inference.",
)
flags.DEFINE_string(
    name="output_format",
    default="csv",
    required=False,
    help="Trajectory output format: 'csv', 'json', or 'both'.",
)
flags.DEFINE_bool(
    name="render_video",
    default=True,
    required=False,
    help="Whether to render an annotated output video.",
)
flags.DEFINE_integer(
    name="trail_length",
    default=30,
    required=False,
    help="Number of past frames for trajectory trail visualization.",
)
flags.DEFINE_integer(
    name="min_track_length",
    default=5,
    required=False,
    help="Minimum number of detections to keep a trajectory.",
)


def main(argv: typing.List[str]) -> int:
    """Main entry point for vehicle trajectory extraction."""
    del argv  # unused console kwargs

    FLAGS = flags.FLAGS
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Step 1: Load model.
    logging.rank_zero_info("Loading YOLO model: %s", FLAGS.model_checkpoint)
    model = tracker.load_model(
        checkpoint_path=FLAGS.model_checkpoint,
        device=FLAGS.device,
    )
    model_name = os.path.splitext(os.path.basename(FLAGS.model_checkpoint))[0]

    # Step 2: Run detection + tracking.
    logging.rank_zero_info("Processing video: %s", FLAGS.video_path)
    trajectory_set = tracker.extract_trajectories(
        model=model,
        video_path=FLAGS.video_path,
        tracker_config=FLAGS.tracker_config,
        confidence_threshold=FLAGS.confidence_threshold,
        iou_threshold=FLAGS.iou_threshold,
        img_size=FLAGS.img_size,
    )

    # Step 3: Filter short trajectories.
    if FLAGS.min_track_length > 0:
        filtered = {
            tid: traj
            for tid, traj in trajectory_set.trajectories.items()
            if len(traj.detections) >= FLAGS.min_track_length
        }
        removed = len(trajectory_set.trajectories) - len(filtered)
        trajectory_set.trajectories = filtered
        logging.rank_zero_info(
            "Filtered %d short trajectories (min_length=%d); %d remain.",
            removed,
            FLAGS.min_track_length,
            len(filtered),
        )

    # Step 4: Save trajectory data.
    video_basename = os.path.splitext(os.path.basename(FLAGS.video_path))[0]

    if FLAGS.output_format in ("csv", "both"):
        csv_path = os.path.join(
            FLAGS.output_dir, f"{video_basename}_trajectories.csv"
        )
        serialization.save_trajectories_csv(trajectory_set, csv_path)

    if FLAGS.output_format in ("json", "both"):
        json_path = os.path.join(
            FLAGS.output_dir,
            f"{model_name}_{video_basename}_trajectories.json",
        )
        serialization.save_trajectories_json(trajectory_set, json_path)

    # Step 5: Render annotated video.
    if FLAGS.render_video:
        annotated_path = os.path.join(
            FLAGS.output_dir,
            f"{model_name}_{video_basename}_annotated.mp4",
        )
        visualization.render_annotated_video(
            trajectory_set=trajectory_set,
            output_path=annotated_path,
            trail_length=FLAGS.trail_length,
        )

    logging.rank_zero_info(
        "Pipeline complete. Outputs in: %s",
        FLAGS.output_dir,
    )
    return 0


if __name__ == "__main__":
    app.run(main=main)
