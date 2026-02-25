import sys

import numpy as np
import pytest

from src.projects.jtrp.experimental import structure


class TestBoundingBox:
    r"""Test cases for `BoundingBox` instances."""

    def test_center(self) -> None:
        bbox = structure.BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
        cx, cy = bbox.center
        assert cx == pytest.approx(20.0)
        assert cy == pytest.approx(40.0)

    def test_width_and_height(self) -> None:
        bbox = structure.BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
        assert bbox.width == pytest.approx(20.0)
        assert bbox.height == pytest.approx(40.0)

    def test_area(self) -> None:
        bbox = structure.BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=5.0)
        assert bbox.area == pytest.approx(50.0)

    def test_zero_size_bbox(self) -> None:
        bbox = structure.BoundingBox(x1=5.0, y1=5.0, x2=5.0, y2=5.0)
        assert bbox.width == pytest.approx(0.0)
        assert bbox.height == pytest.approx(0.0)
        assert bbox.area == pytest.approx(0.0)
        cx, cy = bbox.center
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)


class TestDetection:
    r"""Test cases for `Detection` instances."""

    def test_detection_creation(self) -> None:
        det = structure.Detection(
            frame_index=0,
            track_id=1,
            bbox=structure.BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
            class_id=2,
            class_name="car",
            confidence=0.95,
        )
        assert det.frame_index == 0
        assert det.track_id == 1
        assert det.class_name == "car"
        assert det.confidence == pytest.approx(0.95)


def _make_detection(
    frame_index,
    track_id=1,
    class_name="car",
    conf=0.9,
) -> structure.Detection:
    r"""Helper to create a Detection with sensible defaults."""
    return structure.Detection(
        frame_index=frame_index,
        track_id=track_id,
        bbox=structure.BoundingBox(
            x1=float(frame_index * 10),
            y1=float(frame_index * 10),
            x2=float(frame_index * 10 + 20),
            y2=float(frame_index * 10 + 20),
        ),
        class_id=2,
        class_name=class_name,
        confidence=conf,
    )


class TestTrajectory:
    r"""Test cases for `Trajectory` instances."""

    def test_frame_indices(self) -> None:
        traj = structure.Trajectory(
            track_id=1,
            detections=[
                _make_detection(0),
                _make_detection(1),
                _make_detection(3),
            ],
        )
        assert traj.frame_indices == [0, 1, 3]

    def test_center_positions_shape(self) -> None:
        traj = structure.Trajectory(
            track_id=1,
            detections=[_make_detection(i) for i in range(5)],
        )
        centers = traj.center_positions
        assert isinstance(centers, np.ndarray)
        assert centers.shape == (5, 2)

    def test_center_positions_values(self) -> None:
        traj = structure.Trajectory(
            track_id=1,
            detections=[_make_detection(0)],
        )
        centers = traj.center_positions
        # bbox is (0, 0, 20, 20), center should be (10, 10)
        np.testing.assert_allclose(centers[0], [10.0, 10.0])

    def test_bounding_boxes_shape(self) -> None:
        traj = structure.Trajectory(
            track_id=1,
            detections=[_make_detection(i) for i in range(3)],
        )
        boxes = traj.bounding_boxes
        assert isinstance(boxes, np.ndarray)
        assert boxes.shape == (3, 4)

    def test_confidence_scores(self) -> None:
        traj = structure.Trajectory(
            track_id=1,
            detections=[
                _make_detection(0, conf=0.9),
                _make_detection(1, conf=0.8),
            ],
        )
        assert traj.confidence_scores == [
            pytest.approx(0.9),
            pytest.approx(0.8),
        ]

    def test_dominant_class(self) -> None:
        traj = structure.Trajectory(
            track_id=1,
            detections=[
                _make_detection(0, class_name="car"),
                _make_detection(1, class_name="car"),
                _make_detection(2, class_name="truck"),
            ],
        )
        assert traj.dominant_class == "car"

    def test_dominant_class_tie_breaks_to_most_common(self) -> None:
        traj = structure.Trajectory(
            track_id=1,
            detections=[
                _make_detection(0, class_name="truck"),
                _make_detection(1, class_name="truck"),
                _make_detection(2, class_name="car"),
                _make_detection(3, class_name="car"),
                _make_detection(4, class_name="truck"),
            ],
        )
        assert traj.dominant_class == "truck"


class TestTrajectorySet:
    r"""Test cases for `TrajectorySet` instances."""

    def _make_trajectory_set(self) -> structure.TrajectorySet:
        return structure.TrajectorySet(
            source_video="/tmp/test.mp4",
            frame_width=1920,
            frame_height=1080,
            fps=30.0,
            total_frames=100,
        )

    def test_add_detection_creates_trajectory(self) -> None:
        tset = self._make_trajectory_set()
        with pytest.raises(TypeError):
            tset.add_detection(1)  # type: ignore

        det = _make_detection(0, track_id=1)
        tset.add_detection(det)

        assert 1 in tset.trajectories
        assert len(tset.trajectories[1].detections) == 1

    def test_add_detection_groups_by_track_id(self) -> None:
        tset = self._make_trajectory_set()
        tset.add_detection(_make_detection(0, track_id=1))
        tset.add_detection(_make_detection(1, track_id=1))
        tset.add_detection(_make_detection(0, track_id=2))

        assert len(tset.trajectories) == 2
        assert len(tset.trajectories[1].detections) == 2
        assert len(tset.trajectories[2].detections) == 1

    def test_empty_trajectory_set(self) -> None:
        tset = self._make_trajectory_set()
        assert len(tset.trajectories) == 0

    def test_metadata_preserved(self) -> None:
        tset = self._make_trajectory_set()
        assert tset.source_video == "/tmp/test.mp4"
        assert tset.frame_width == 1920
        assert tset.frame_height == 1080
        assert tset.fps == pytest.approx(30.0)
        assert tset.total_frames == 100


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
