import csv
import json
import os
import sys
import tempfile

import numpy as np
import pytest

from src.projects.jtrp.rci import georeferencing
from src.projects.jtrp.rci import serialization
from src.projects.jtrp.rci import structure

# Constants
TMP_DIR = tempfile.gettempdir()


def _make_trajectory_set() -> structure.TrajectorySet:
    r"""Creates a ``TrajectorySet`` with known data for round-trip tests."""
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
                    "world_x",
                    "world_y",
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
            assert loaded_det.world_x == orig_det.world_x
            assert loaded_det.world_y == orig_det.world_y


class TestWorldCoordSerialization:
    r"""Tests for the world_x/world_y fields added for georeferencing."""

    def _make_set_with_world(self) -> structure.TrajectorySet:
        tset = structure.TrajectorySet(
            source_video=os.path.join(TMP_DIR, "world.mp4"),
            frame_width=1920,
            frame_height=1080,
            fps=30.0,
            total_frames=10,
        )
        for i in range(3):
            tset.add_detection(
                structure.Detection(
                    frame_index=i,
                    track_id=1,
                    bbox=structure.BoundingBox(
                        x1=100.0 + i * 10,
                        y1=200.0,
                        x2=150.0 + i * 10,
                        y2=300.0,
                    ),
                    class_id=2,
                    class_name="car",
                    confidence=0.9,
                    world_x=237_000.0 + i * 5.0,
                    world_y=1_753_000.0 + i * 0.5,
                )
            )
        return tset

    def test_csv_includes_world_values(self) -> None:
        tset = self._make_set_with_world()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "world.csv")
            serialization.save_trajectories_csv(tset, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        assert len(rows) == 3
        for row in rows:
            assert row["world_x"] != ""
            assert row["world_y"] != ""
        assert float(rows[0]["world_x"]) == pytest.approx(237_000.0, abs=0.001)

    def test_csv_empty_world_for_missing(self) -> None:
        tset = structure.TrajectorySet(
            source_video=os.path.join(TMP_DIR, "noworld.mp4"),
            frame_width=10,
            frame_height=10,
            fps=10.0,
            total_frames=1,
        )
        tset.add_detection(
            structure.Detection(
                frame_index=0,
                track_id=1,
                bbox=structure.BoundingBox(0.0, 0.0, 1.0, 1.0),
                class_id=2,
                class_name="car",
                confidence=0.5,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "noworld.csv")
            serialization.save_trajectories_csv(tset, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        assert rows[0]["world_x"] == ""
        assert rows[0]["world_y"] == ""

    def test_json_round_trip_with_world(self) -> None:
        tset = self._make_set_with_world()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "world.json")
            serialization.save_trajectories_json(tset, path)
            loaded = serialization.load_trajectories_json(path)
        loaded_dets = loaded.trajectories[1].detections
        orig_dets = tset.trajectories[1].detections
        for orig, got in zip(orig_dets, loaded_dets):
            assert got.world_x == pytest.approx(orig.world_x)
            assert got.world_y == pytest.approx(orig.world_y)

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


class TestCameraParametersSerialization:
    r"""Round-trip tests for ``structure.CameraParameters``."""

    def _make_params(self) -> structure.CameraParameters:
        return structure.CameraParameters(
            camera_matrix=np.array(
                [[1000.0, 0.0, 480.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]
            ),
            dist_coeffs=np.array([-0.1, 0.05, 0.0, 0.0, 0.0]),
            img_size=(960, 720),
            rms_reprojection_error=0.42,
            pattern_size=(7, 6),
            square_size=0.025,
            num_views=12,
        )

    def test_round_trip(self) -> None:
        params = self._make_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "calib.json")
            serialization.save_camera_parameters(params, path)
            loaded = serialization.load_camera_parameters(path)

        np.testing.assert_allclose(loaded.camera_matrix, params.camera_matrix)
        np.testing.assert_allclose(loaded.dist_coeffs, params.dist_coeffs)
        assert loaded.img_size == params.img_size
        assert loaded.rms_reprojection_error == pytest.approx(
            params.rms_reprojection_error
        )
        assert loaded.pattern_size == params.pattern_size
        assert loaded.square_size == pytest.approx(params.square_size)
        assert loaded.num_views == params.num_views

    def test_save_creates_parent_dir(self) -> None:
        params = self._make_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "subdir", "calib.json")
            serialization.save_camera_parameters(params, path)
            assert os.path.isfile(path)


def _square_gcps(
    image_size: int = 1000,
    world_origin_x: float = 1_000_000.0,
    world_origin_y: float = 2_000_000.0,
    world_extent: float = 100.0,
):
    return [
        structure.GroundControlPoint(
            label="1",
            world_x=world_origin_x,
            world_y=world_origin_y,
            image_u=0.0,
            image_v=float(image_size),
        ),
        structure.GroundControlPoint(
            label="2",
            world_x=world_origin_x + world_extent,
            world_y=world_origin_y,
            image_u=float(image_size),
            image_v=float(image_size),
        ),
        structure.GroundControlPoint(
            label="3",
            world_x=world_origin_x + world_extent,
            world_y=world_origin_y + world_extent,
            image_u=float(image_size),
            image_v=0.0,
        ),
        structure.GroundControlPoint(
            label="4",
            world_x=world_origin_x,
            world_y=world_origin_y + world_extent,
            image_u=0.0,
            image_v=0.0,
        ),
    ]


class TestGcpJsonSerialization:
    r"""Round-trip tests for the GCP JSON file format."""

    def test_round_trip(self) -> None:
        gcps = _square_gcps()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gcps.json")
            serialization.save_gcps_to_json(gcps, path)
            loaded = serialization.load_gcps_from_json(path)
        assert len(loaded) == len(gcps)
        for a, b in zip(gcps, loaded):
            assert a.label == b.label
            assert a.world_x == pytest.approx(b.world_x)
            assert a.world_y == pytest.approx(b.world_y)
            assert a.image_u == pytest.approx(b.image_u)
            assert a.image_v == pytest.approx(b.image_v)

    def test_metadata_is_preserved(self) -> None:
        gcps = _square_gcps()
        meta = {"world_units": "ftus", "source_csv": "/foo/bar.csv"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gcps.json")
            serialization.save_gcps_to_json(gcps, path, metadata=meta)
            with open(path) as f:
                payload = json.load(f)
        assert payload["metadata"] == meta

    def test_load_raises_when_gcps_field_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "broken.json")
            with open(path, "w") as f:
                json.dump({"nope": []}, f)
            with pytest.raises(ValueError):
                serialization.load_gcps_from_json(path)


class TestGeoReferenceSerialization:
    r"""Round-trip tests for ``structure.GeoReference``."""

    def test_round_trip(self) -> None:
        geo = georeferencing.compute_homography(_square_gcps())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "geo.json")
            serialization.save_georeference(geo, path)
            loaded = serialization.load_georeference(path)
        np.testing.assert_allclose(
            loaded.homography_pix_to_world, geo.homography_pix_to_world
        )
        np.testing.assert_allclose(
            loaded.homography_world_to_pix, geo.homography_world_to_pix
        )
        assert loaded.world_units == geo.world_units
        assert loaded.world_crs == geo.world_crs
        assert loaded.rms_reprojection_error_px == pytest.approx(
            geo.rms_reprojection_error_px
        )
        assert len(loaded.gcps) == len(geo.gcps)


class TestImageUvCsv:
    r"""Round-trip tests for the ``label,u,v,area_px`` marker CSV."""

    def test_round_trip(self) -> None:
        markers = [
            ("1", 100.0, 200.0, 1234.0),
            ("2", 150.5, 220.25, 567.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "uv.csv")
            serialization.save_image_uv_csv(markers, path)
            loaded = serialization.load_image_uv_csv(path)
        assert loaded["1"] == pytest.approx((100.0, 200.0))
        assert loaded["2"] == pytest.approx((150.5, 220.25))

    def test_load_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "broken.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["label", "u"])  # missing 'v'
                writer.writerow(["1", "10"])
            with pytest.raises(ValueError):
                serialization.load_image_uv_csv(path)


class TestLoadWorldCsv:
    r"""Tests for the INDOT-format world CSV reader."""

    def test_parses_simple_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gcps.csv")
            with open(path, "w") as f:
                f.write(
                    "Point Number,Northing (ft (US)),Easting (ft (US)),"
                    "Orthometric height (ft (US))\n"
                )
                f.write("1,1753019.747,237379.900,782.045\n")
                f.write("2,1752943.849,237383.549,782.287\n")
            rows = serialization.load_world_csv(path)
        assert len(rows) == 2
        assert rows[0][0] == "1"
        # world_x = Easting, world_y = Northing.
        assert rows[0][1] == pytest.approx(237379.900)
        assert rows[0][2] == pytest.approx(1753019.747)

    def test_table_index_selects_specific_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gcps.csv")
            with open(path, "w") as f:
                f.write(
                    "Point Number,Northing (ft (US)),Easting (ft (US)),"
                    "Orthometric height (ft (US))\n"
                )
                f.write("1,1.0,2.0,3.0\n")
                f.write(
                    "Point Number,Northing (ft (US)),Easting (ft (US)),"
                    "Orthometric height (ft (US))\n"
                )
                f.write("1,10.0,20.0,30.0\n")
            first = serialization.load_world_csv(path, table_index=1)
            second = serialization.load_world_csv(path, table_index=2)
        assert len(first) == 1 and first[0][1] == pytest.approx(2.0)
        assert len(second) == 1 and second[0][1] == pytest.approx(20.0)

    def test_table_index_out_of_range_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gcps.csv")
            with open(path, "w") as f:
                f.write(
                    "Point Number,Northing (ft (US)),Easting (ft (US)),"
                    "Orthometric height (ft (US))\n"
                )
                f.write("1,1.0,2.0,3.0\n")
            with pytest.raises(ValueError):
                serialization.load_world_csv(path, table_index=2)

    def test_zero_table_index_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gcps.csv")
            with open(path, "w") as f:
                f.write(
                    "Point Number,Northing (ft (US)),Easting (ft (US)),"
                    "Orthometric height (ft (US))\n"
                )
                f.write("1,1.0,2.0,3.0\n")
            with pytest.raises(ValueError):
                serialization.load_world_csv(path, table_index=0)

    def test_missing_header_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gcps.csv")
            with open(path, "w") as f:
                f.write("foo,bar\n1,2\n")
            with pytest.raises(ValueError):
                serialization.load_world_csv(path)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
