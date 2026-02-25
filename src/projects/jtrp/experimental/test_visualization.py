import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

from src.projects.jtrp.experimental import structure
from src.projects.jtrp.experimental import visualization


def _create_synthetic_video(
    path,
    width=320,
    height=240,
    fps=10.0,
    num_frames=30,
) -> None:
    r"""Creates a small synthetic video (solid color frames) for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Vary the shade so frames are distinguishable.
        shade = int(128 + 4 * i) % 256
        frame = np.full((height, width, 3), shade, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_trajectory_set(
    video_path,
    width=320,
    height=240,
    fps=10.0,
    num_frames=30,
) -> structure.TrajectorySet:
    r"""Creates a `TrajectorySet` with detections over a synthetic video."""
    tset = structure.TrajectorySet(
        source_video=video_path,
        frame_width=width,
        frame_height=height,
        fps=fps,
        total_frames=num_frames,
    )

    # Add track 1 across some frames.
    for i in range(min(10, num_frames)):
        tset.add_detection(
            structure.Detection(
                frame_index=i,
                track_id=1,
                bbox=structure.BoundingBox(
                    x1=50.0 + i * 5,
                    y1=50.0 + i * 3,
                    x2=100.0 + i * 5,
                    y2=100.0 + i * 3,
                ),
                class_id=2,
                class_name="car",
                confidence=0.92,
            )
        )

    # Add track 2 across fewer frames.
    for i in range(min(5, num_frames)):
        tset.add_detection(
            structure.Detection(
                frame_index=i,
                track_id=2,
                bbox=structure.BoundingBox(
                    x1=200.0 - i * 3,
                    y1=100.0 + i * 2,
                    x2=260.0 - i * 3,
                    y2=160.0 + i * 2,
                ),
                class_id=7,
                class_name="truck",
                confidence=0.85,
            )
        )

    return tset


class TestGetColor:
    r"""Unit tests for `_get_color` function."""

    def test_deterministic(self) -> None:
        assert visualization._get_color(5) == visualization._get_color(5)

    def test_different_ids_get_different_colors(self) -> None:
        colors = {visualization._get_color(i) for i in range(12)}
        assert len(colors) == 12

    def test_wraps_around(self) -> None:
        assert visualization._get_color(0) == visualization._get_color(12)


class TestRenderAnnotatedVideo:
    r"""Unit tests for `render_annotated_video` function."""

    def test_output_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "annotated.mp4")
            _create_synthetic_video(video_path)

            tset = _make_trajectory_set(video_path)
            visualization.render_annotated_video(tset, output_path)

            assert os.path.isfile(output_path)

    def test_output_frame_count(self):
        num_frames = 20
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "annotated.mp4")
            _create_synthetic_video(video_path, num_frames=num_frames)

            tset = _make_trajectory_set(video_path, num_frames=num_frames)
            visualization.render_annotated_video(tset, output_path)

            cap = cv2.VideoCapture(output_path)
            out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            assert out_frames == num_frames

    def test_output_dimensions(self):
        width, height = 320, 240
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "annotated.mp4")
            _create_synthetic_video(video_path, width=width, height=height)

            tset = _make_trajectory_set(video_path, width=width, height=height)
            visualization.render_annotated_video(tset, output_path)

            cap = cv2.VideoCapture(output_path)
            out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            assert out_w == width
            assert out_h == height

    def test_no_detections_produces_valid_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "annotated.mp4")
            _create_synthetic_video(video_path, num_frames=10)

            tset = structure.TrajectorySet(
                source_video=video_path,
                frame_width=320,
                frame_height=240,
                fps=10.0,
                total_frames=10,
            )
            visualization.render_annotated_video(tset, output_path)

            assert os.path.isfile(output_path)
            cap = cv2.VideoCapture(output_path)
            out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            assert out_frames == 10

    def test_bbox_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "annotated.mp4")
            _create_synthetic_video(video_path, num_frames=5)

            tset = _make_trajectory_set(video_path, num_frames=5)
            visualization.render_annotated_video(
                tset,
                output_path,
                show_bbox=True,
                show_label=False,
                show_trail=False,
            )

            assert os.path.isfile(output_path)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
