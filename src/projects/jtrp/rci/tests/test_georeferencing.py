import sys

import numpy as np
import pytest

from src.projects.jtrp.rci import georeferencing
from src.projects.jtrp.rci import structure


def _square_gcps(
    image_size: int = 1000,
    world_origin_x: float = 1_000_000.0,
    world_origin_y: float = 2_000_000.0,
    world_extent: float = 100.0,
):
    r"""Creates 4 GCPs at the corners of a known rectangle in both frames."""
    return [
        structure.GroundControlPoint(
            label="1",
            world_x=world_origin_x + 0.0,
            world_y=world_origin_y + 0.0,
            image_u=0.0,
            image_v=float(image_size),
        ),
        structure.GroundControlPoint(
            label="2",
            world_x=world_origin_x + world_extent,
            world_y=world_origin_y + 0.0,
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
            world_x=world_origin_x + 0.0,
            world_y=world_origin_y + world_extent,
            image_u=0.0,
            image_v=0.0,
        ),
    ]


class TestComputeHomography:
    r"""Unit tests for the ``compute_homography`` function."""

    def test_recovers_known_transform(self) -> None:
        r"""Test that a perfect homography is recovered from ideal GCPs."""

        gcps = _square_gcps()
        geo = georeferencing.compute_homography(gcps)
        assert geo.homography_pix_to_world.shape == (3, 3)
        assert geo.rms_reprojection_error_px == pytest.approx(0.0, abs=1e-6)

    def test_pixel_to_world_roundtrip(self) -> None:
        r"""Test inverse consistency of pixel_to_world and world_to_pixel."""

        gcps = _square_gcps()
        geo = georeferencing.compute_homography(gcps)
        u, v = 500.0, 500.0
        x, y = georeferencing.pixel_to_world(geo, u, v)
        u2, v2 = georeferencing.world_to_pixel(geo, x, y)
        assert u2 == pytest.approx(u, abs=1e-6)
        assert v2 == pytest.approx(v, abs=1e-6)

    def test_center_pixel_maps_to_center_world(self) -> None:
        r"""Test that the center pixel maps to the center world coordinate."""

        gcps = _square_gcps()
        geo = georeferencing.compute_homography(gcps)
        # Image center (500, 500) -> world (1_000_050, 2_000_050).
        x, y = georeferencing.pixel_to_world(geo, 500.0, 500.0)
        assert x == pytest.approx(1_000_050.0, abs=1e-3)
        assert y == pytest.approx(2_000_050.0, abs=1e-3)

    def test_raises_on_fewer_than_four_gcps(self) -> None:
        r"""Test that a ValueError is raised if given less than four GCPs."""

        gcps = _square_gcps()[:3]
        with pytest.raises(ValueError):
            georeferencing.compute_homography(gcps)

    def test_transform_points_vectorized(self) -> None:
        r"""Test consistency between two methods to apply the homography."""

        gcps = _square_gcps()
        geo = georeferencing.compute_homography(gcps)
        pts = np.array([[0.0, 1000.0], [1000.0, 0.0], [500.0, 500.0]])
        out = georeferencing.transform_points(pts, geo.homography_pix_to_world)
        # Match per-element pixel_to_world calls.
        for p, ref in zip(pts, out):
            x, y = georeferencing.pixel_to_world(geo, p[0], p[1])
            assert x == pytest.approx(ref[0])
            assert y == pytest.approx(ref[1])


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
