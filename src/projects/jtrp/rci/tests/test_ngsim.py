import csv
import os
import sys
import tempfile

import numpy as np
import pytest

from src.projects.jtrp.rci import ngsim
from src.projects.jtrp.rci import structure

# -- constants -----------------------------------------------------------------
TMP_DIR = tempfile.gettempdir()


def _make_linear_trajectory(
    track_id: int = 1,
    n_frames: int = 30,
    fps: float = 30.0,
    start_world: tuple = (1000.0, 2000.0),
    velocity_per_frame: tuple = (1.0, 0.0),
    class_id: int = 2,
    class_name: str = "car",
) -> structure.TrajectorySet:
    r"""Creates an example ``TrajectorySet`` with a single linear track."""
    tset = structure.TrajectorySet(
        source_video=os.path.join(TMP_DIR, "dummy.mp4"),
        frame_width=1920,
        frame_height=1080,
        fps=fps,
        total_frames=n_frames,
    )
    for i in range(n_frames):
        wx = start_world[0] + velocity_per_frame[0] * i
        wy = start_world[1] + velocity_per_frame[1] * i
        tset.add_detection(
            structure.Detection(
                frame_index=i,
                track_id=track_id,
                bbox=structure.BoundingBox(
                    x1=100.0 + i,
                    y1=100.0,
                    x2=120.0 + i,
                    y2=140.0,
                ),
                class_id=class_id,
                class_name=class_name,
                confidence=0.9,
                world_x=wx,
                world_y=wy,
            )
        )
    return tset


class TestSaveNgsimCsv:
    r"""Unit tests for the ``save_ngsim_csv`` function."""

    def test_columns_match_ngsim_spec(self) -> None:
        r"""Test the output has the expected columns in the correct order."""

        tset = _make_linear_trajectory()
        cfg = ngsim.NgsimExportConfig(location="UnitTest")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                reader = csv.DictReader(f)
                assert (
                    tuple(reader.fieldnames or ())  # type: ignore
                    == ngsim.NGSIM_FIELDNAMES
                )

    def test_row_count_matches_detections(self) -> None:
        r"""Test the output has one row per detection (after filtering)."""

        tset = _make_linear_trajectory(n_frames=30)
        cfg = ngsim.NgsimExportConfig(location="UnitTest")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        assert len(rows) == 30

    def test_global_time_increments_with_frame(self) -> None:
        r"""Test ``Global_Time`` increments along the frame index and fps."""

        tset = _make_linear_trajectory(n_frames=6, fps=10.0)
        cfg = ngsim.NgsimExportConfig(
            location="UnitTest", global_time_zero_ms=1_000
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        # fps=10 -> 100 ms/frame; t0=1000ms.
        assert int(rows[0]["Global_Time"]) == 1000
        assert int(rows[1]["Global_Time"]) == 1100
        assert int(rows[2]["Global_Time"]) == 1200

    def test_velocity_recovers_linear_motion(self) -> None:
        r"""Test the output velocities match the known linear motion."""

        # World units: 1 unit / frame at 30 fps -> 30 units / second.
        tset = _make_linear_trajectory(
            n_frames=20,
            fps=30.0,
            velocity_per_frame=(1.0, 0.0),
        )
        cfg = ngsim.NgsimExportConfig(location="UnitTest", smoothing_window=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        speeds = [float(r["v_Vel"]) for r in rows[1:-1]]  # interior points
        assert all(abs(s - 30.0) < 1e-6 for s in speeds)
        accs = [float(r["v_Acc"]) for r in rows[2:-2]]
        assert all(abs(a) < 1e-6 for a in accs)

    def test_local_origin_default_to_min_world(self) -> None:
        r"""Test the default local origin is the minimum world coordinates."""

        tset = _make_linear_trajectory(
            start_world=(1000.0, 2000.0),
            velocity_per_frame=(1.0, 0.0),
            n_frames=10,
        )
        cfg = ngsim.NgsimExportConfig(location="UnitTest", smoothing_window=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        # First row should have Local_X = 0 (since smoothing=0 the
        # smoothed start is exactly the first world point).
        assert float(rows[0]["Local_X"]) == pytest.approx(0.0, abs=1e-6)
        assert float(rows[0]["Local_Y"]) == pytest.approx(0.0, abs=1e-6)

    def test_explicit_local_origin(self) -> None:
        r"""Test an explicit local origin is correctly applied."""

        tset = _make_linear_trajectory(
            start_world=(1000.0, 2000.0),
            n_frames=5,
            velocity_per_frame=(1.0, 0.0),
        )
        cfg = ngsim.NgsimExportConfig(
            location="UnitTest",
            local_origin=(500.0, 1500.0),
            smoothing_window=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        assert float(rows[0]["Local_X"]) == pytest.approx(500.0)
        assert float(rows[0]["Local_Y"]) == pytest.approx(500.0)

    def test_short_trajectory_filtered(self) -> None:
        r"""Test a trajectory shorter than the minimum length is filtered."""

        tset = _make_linear_trajectory(n_frames=3)
        cfg = ngsim.NgsimExportConfig(location="UnitTest", min_track_length=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            with pytest.raises(ValueError):
                ngsim.save_ngsim_csv(tset, path, cfg)

    def test_short_trajectory_kept_when_filter_disabled(self) -> None:
        r"""Test a short trajectory is kept when filtering is disabled."""

        tset = _make_linear_trajectory(n_frames=3)
        cfg = ngsim.NgsimExportConfig(location="UnitTest", emit_filtered=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_missing_world_raises(self) -> None:
        r"""Test a detection with missing world coordinates raises an error."""

        tset = _make_linear_trajectory(n_frames=10)
        # Strip world coords from one detection.
        tset.trajectories[1].detections[0].world_x = None
        tset.trajectories[1].detections[0].world_y = None
        cfg = ngsim.NgsimExportConfig(location="UnitTest")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            with pytest.raises(ValueError):
                ngsim.save_ngsim_csv(tset, path, cfg)

    def test_class_id_mapping(self) -> None:
        r"""Test COCO class IDs are mapped to the correct NGSIM `v_classes`."""
        # Build a trajectory for each known COCO class.
        tset = structure.TrajectorySet(
            source_video=os.path.join(TMP_DIR, "dummy.mp4"),
            frame_width=10,
            frame_height=10,
            fps=10.0,
            total_frames=10,
        )
        for tid, (cls_id, cls_name) in enumerate(
            [(2, "car"), (3, "motorcycle"), (5, "bus"), (7, "truck")],
            start=1,
        ):
            for i in range(6):
                tset.add_detection(
                    structure.Detection(
                        frame_index=i,
                        track_id=tid,
                        bbox=structure.BoundingBox(0.0, 0.0, 5.0, 5.0),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=0.9,
                        world_x=float(i),
                        world_y=0.0,
                    )
                )
        cfg = ngsim.NgsimExportConfig(location="UnitTest")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        # Vehicle_ID == 1 (car) -> v_Class 2 (auto).
        vid_to_class = {int(r["Vehicle_ID"]): int(r["v_Class"]) for r in rows}
        assert vid_to_class[1] == 2  # car
        assert vid_to_class[2] == 1  # motorcycle
        assert vid_to_class[3] == 4  # bus
        assert vid_to_class[4] == 3  # truck

    def test_rows_sorted_by_vehicle_then_frame(self) -> None:
        tset = structure.TrajectorySet(
            source_video=os.path.join(TMP_DIR, "dummy.mp4"),
            frame_width=10,
            frame_height=10,
            fps=10.0,
            total_frames=10,
        )
        # Add detections out of order.
        for tid in (3, 1, 2):
            for i in (5, 1, 3, 0):
                tset.add_detection(
                    structure.Detection(
                        frame_index=i,
                        track_id=tid,
                        bbox=structure.BoundingBox(0.0, 0.0, 5.0, 5.0),
                        class_id=2,
                        class_name="car",
                        confidence=0.5,
                        world_x=float(i),
                        world_y=0.0,
                    )
                )
        cfg = ngsim.NgsimExportConfig(
            location="UnitTest",
            emit_filtered=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ngsim.csv")
            ngsim.save_ngsim_csv(tset, path, cfg)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        keys = [(int(r["Vehicle_ID"]), int(r["Frame_ID"])) for r in rows]
        assert keys == sorted(keys)


class TestMovingAverage:
    r"""Unit tests for the ``_moving_average`` function."""

    def test_zero_window_returns_copy(self) -> None:
        r"""Test that a zero smoothing window returns a copy of the input."""

        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = ngsim._moving_average(x, 0)
        np.testing.assert_allclose(out, x)
        # Confirms a copy, not the same object.
        assert out is not x

    def test_window_one_smooths(self) -> None:
        r"""Test that a window of ``1`` averages over the last 3 points."""

        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        out = ngsim._moving_average(x, 1)
        # i=0: avg(0,1)=0.5; i=1: avg(0,1,2)=1; i=2: avg(1,2,3)=2;
        # i=3: avg(2,3)=2.5
        np.testing.assert_allclose(out, [[0.5], [1.0], [2.0], [2.5]])


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
