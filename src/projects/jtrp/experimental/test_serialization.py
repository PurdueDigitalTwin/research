import csv
import json
import os
import sys
import tempfile

import pytest

from src.projects.jtrp.experimental import serialization
from src.projects.jtrp.experimental import structure

# Constants
TMP_DIR = tempfile.gettempdir()


def _make_trajectory_set() -> structure.TrajectorySet:
    r"""Creates a `TrajectorySet` with known data for round-trip tests."""
    tset = structure.TrajectorySet(
        source_video=os.path.join(TMP_DIR, "test_video.mp4"),
        frame_width=1920,
        frame_height=1080,
        fps=30.0,
        total_frames=100,
    )

    # Track 1: car across 3 frames.
    for i in range(3):
        tset.add_detection(
            structure.Detection(
                frame_index=i,
                track_id=1,
                bbox=structure.BoundingBox(
                    x1=100.0 + i * 10,
                    y1=200.0 + i * 5,
                    x2=150.0 + i * 10,
                    y2=250.0 + i * 5,
                ),
                class_id=2,
                class_name="car",
                confidence=0.95 - i * 0.01,
            )
        )

    # Track 2: truck across 2 frames.
    for i in range(2):
        tset.add_detection(
            structure.Detection(
                frame_index=i,
                track_id=2,
                bbox=structure.BoundingBox(
                    x1=500.0 + i * 20,
                    y1=300.0 + i * 10,
                    x2=600.0 + i * 20,
                    y2=400.0 + i * 10,
                ),
                class_id=7,
                class_name="truck",
                confidence=0.88 - i * 0.02,
            )
        )

    return tset


class TestCsvSerialization:
    r"""Unit tests for CSV serialization functions."""

    def test_csv_output_has_expected_columns(self) -> None:
        tset = _make_trajectory_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "traj.csv")
            serialization.save_trajectories_csv(tset, path)

            with open(path) as f:
                reader = csv.DictReader(f)
                assert set(reader.fieldnames) == {  # type: ignore
                    "frame_index",
                    "track_id",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "cx",
                    "cy",
                    "class_id",
                    "class_name",
                    "confidence",
                }

    def test_csv_row_count(self) -> None:
        tset = _make_trajectory_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "traj.csv")
            serialization.save_trajectories_csv(tset, path)

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # 3 detections for track 1 + 2 for track 2 = 5 total
            assert len(rows) == 5

    def test_csv_is_sorted_by_frame_then_track(self) -> None:
        tset = _make_trajectory_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "traj.csv")
            serialization.save_trajectories_csv(tset, path)

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            keys = [(int(r["frame_index"]), int(r["track_id"])) for r in rows]
            assert keys == sorted(keys)

    def test_csv_center_values(self) -> None:
        tset = _make_trajectory_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "traj.csv")
            serialization.save_trajectories_csv(tset, path)

            with open(path) as f:
                reader = csv.DictReader(f)
                first_row = next(reader)

            # First detection: bbox (100, 200, 150, 250), center (125, 225)
            assert float(first_row["cx"]) == pytest.approx(125.0, abs=0.01)
            assert float(first_row["cy"]) == pytest.approx(225.0, abs=0.01)

    def test_csv_empty_trajectory_set(self) -> None:
        tset = structure.TrajectorySet(
            source_video=os.path.join(TMP_DIR, "empty.mp4"),
            frame_width=1920,
            frame_height=1080,
            fps=30.0,
            total_frames=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.csv")
            serialization.save_trajectories_csv(tset, path)

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 0


class TestJsonSerialization:
    r"""Unit tests for JSON serialization functions."""

    def test_json_metadata(self) -> None:
        tset = _make_trajectory_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "traj.json")
            serialization.save_trajectories_json(tset, path)

            with open(path) as f:
                data = json.load(f)

            meta = data["metadata"]
            assert meta["source_video"] == os.path.join(
                TMP_DIR,
                "test_video.mp4",
            )
            assert meta["frame_width"] == 1920
            assert meta["frame_height"] == 1080
            assert meta["fps"] == pytest.approx(30.0)
            assert meta["total_frames"] == 100
            assert meta["num_trajectories"] == 2

    def test_json_trajectory_count(self) -> None:
        tset = _make_trajectory_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "traj.json")
            serialization.save_trajectories_json(tset, path)

            with open(path) as f:
                data = json.load(f)

            assert len(data["trajectories"]) == 2

    def test_json_round_trip(self) -> None:
        tset = _make_trajectory_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "traj.json")
            serialization.save_trajectories_json(tset, path)
            loaded = serialization.load_trajectories_json(path)

        # Verify metadata.
        assert loaded.source_video == tset.source_video
        assert loaded.frame_width == tset.frame_width
        assert loaded.frame_height == tset.frame_height
        assert loaded.fps == pytest.approx(tset.fps)
        assert loaded.total_frames == tset.total_frames

        # Verify trajectory count.
        assert len(loaded.trajectories) == len(tset.trajectories)

        # Verify detection data for track 1.
        orig_t1 = tset.trajectories[1]
        loaded_t1 = loaded.trajectories[1]
        assert len(loaded_t1.detections) == len(orig_t1.detections)

        for orig_det, loaded_det in zip(
            orig_t1.detections, loaded_t1.detections
        ):
            assert loaded_det.frame_index == orig_det.frame_index
            assert loaded_det.track_id == orig_det.track_id
            assert loaded_det.bbox.x1 == pytest.approx(orig_det.bbox.x1)
            assert loaded_det.bbox.y1 == pytest.approx(orig_det.bbox.y1)
            assert loaded_det.bbox.x2 == pytest.approx(orig_det.bbox.x2)
            assert loaded_det.bbox.y2 == pytest.approx(orig_det.bbox.y2)
            assert loaded_det.class_id == orig_det.class_id
            assert loaded_det.class_name == orig_det.class_name
            assert loaded_det.confidence == pytest.approx(orig_det.confidence)

    def test_json_empty_trajectory_set(self):
        tset = structure.TrajectorySet(
            source_video=os.path.join(TMP_DIR, "empty.mp4"),
            frame_width=1920,
            frame_height=1080,
            fps=30.0,
            total_frames=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.json")
            serialization.save_trajectories_json(tset, path)
            loaded = serialization.load_trajectories_json(path)

        assert len(loaded.trajectories) == 0
        assert loaded.source_video == os.path.join(TMP_DIR, "empty.mp4")


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
