import os
import sys
import tempfile
import typing

import cv2
from numpy import typing as npt
import numpy as np
import pytest

from src.projects.jtrp.rci import calibration


@pytest.fixture(scope="module")
def synth_cb_img() -> typing.Callable:
    r"""Renders a synthetic chessboard image on a white background.

    The chessboard has ``cols + 1`` and ``rows + 1`` total squares (so
    ``cols x rows`` interior corners), matching the OpenCV convention.
    """

    def _make_cb_img(
        pattern_size: typing.Tuple[int, int] = (7, 6),
        square_px: int = 60,
        margin: int = 60,
        image_size: typing.Tuple[int, int] = (960, 720),
    ) -> npt.NDArray[np.uint8]:
        cols, rows = pattern_size
        canvas = np.full(
            (image_size[1], image_size[0], 3), 220, dtype=np.uint8
        )
        total_cols, total_rows = cols + 1, rows + 1
        board_w = total_cols * square_px
        board_h = total_rows * square_px
        x0 = margin
        y0 = margin
        if x0 + board_w > image_size[0] or y0 + board_h > image_size[1]:
            raise ValueError(
                "Chessboard does not fit in image_size with margins."
            )

        for r in range(total_rows):
            for c in range(total_cols):
                color = 0 if (r + c) % 2 == 0 else 255
                xa = x0 + c * square_px
                ya = y0 + r * square_px
                xb = xa + square_px
                yb = ya + square_px
                canvas[ya:yb, xa:xb] = color
        return canvas

    return _make_cb_img


@pytest.fixture(scope="module")
def synth_views(synth_cb_img: typing.Callable) -> typing.Callable:
    r"""Creates ``n`` slightly-perturbed renderings of a chessboard."""

    def _make_views(
        pattern_size: typing.Tuple[int, int] = (7, 6),
        n: int = 6,
    ) -> typing.List[npt.NDArray[np.uint8]]:
        base = synth_cb_img(pattern_size, square_px=60)
        views: typing.List[np.ndarray] = []
        h, w = base.shape[:2]
        for i in range(n):
            # Apply a small affine warp so the corner geometry varies between
            # views; this is required for a non-degenerate calibration.
            dx = (i - n / 2) * 8.0
            dy = (i - n / 2) * 5.0
            s = 1.0 + 0.02 * (i - n / 2)
            M = np.array(
                [
                    [s, 0.0, dx],
                    [0.0, s, dy],
                ],
                dtype=np.float64,
            )
            warped = cv2.warpAffine(
                base,
                M,
                (w, h),
                borderValue=(220, 220, 220),
            )
            views.append(warped)
        return views

    return _make_views


class TestSerialization:
    r"""Unit tests for I/O functions of camera parameters."""

    def test_save_and_restore(self) -> None:
        params = calibration.CameraParameters(
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
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "calib.json")
            calibration.json_serialization(params, path)
            loaded = calibration.json_restore(path)

        np.testing.assert_allclose(loaded.camera_matrix, params.camera_matrix)
        np.testing.assert_allclose(loaded.dist_coeffs, params.dist_coeffs)
        assert loaded.img_size == pytest.approx(params.img_size)
        assert loaded.rms_reprojection_error == pytest.approx(
            params.rms_reprojection_error
        )
        assert loaded.pattern_size == params.pattern_size
        assert loaded.square_size == pytest.approx(params.square_size)
        assert loaded.num_views == params.num_views


class TestDetectChessboardCorners:
    r"""Unit tests for chessboard corner detection."""

    def test_detects_synthetic_pattern(self, synth_cb_img) -> None:
        gray = cv2.cvtColor(synth_cb_img(), cv2.COLOR_BGR2GRAY)
        corners = calibration.detect_chessboard_corners(gray, (7, 6))
        assert corners is not None
        assert corners.shape == (7 * 6, 1, 2)

    def test_returns_none_for_non_chessboard(self) -> None:
        gray = np.full((480, 640), 128, dtype=np.uint8)
        corners = calibration.detect_chessboard_corners(gray, (7, 6))
        assert corners is None

    def test_accepts_color_input(self, synth_cb_img) -> None:
        corners = calibration.detect_chessboard_corners(synth_cb_img(), (7, 6))
        assert corners is not None


class TestAutoDetectPatternSize:
    r"""Unit tests for automatic pattern size detection function."""

    def test_picks_correct_pattern(self, synth_views) -> None:
        pattern, count = calibration.auto_detect_pattern_size(
            synth_views(),
            candidates=[(5, 5), (6, 6), (7, 6), (9, 6)],
            min_views=3,
        )
        assert pattern == (7, 6)
        assert count >= 3

    def test_raises_when_no_pattern_detects(self) -> None:
        blanks = [np.full((480, 640), 128, dtype=np.uint8) for _ in range(3)]
        with pytest.raises(ValueError):
            calibration.auto_detect_pattern_size(
                blanks,
                candidates=[(7, 6)],
                min_views=2,
            )

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            calibration.auto_detect_pattern_size([])


class TestCalibrateFromImages:
    r"""Unit tests for camera calibration from chessboard images."""

    def test_runs_on_synthetic_views(self, synth_views) -> None:
        views = synth_views((7, 6), n=8)
        params = calibration.calibrate_from_images(
            views, (7, 6), square_size=0.025
        )
        assert params.camera_matrix.shape == (3, 3)
        assert params.dist_coeffs.ndim == 1
        assert params.img_size == (960, 720)
        assert params.pattern_size == (7, 6)
        assert params.square_size == pytest.approx(0.025)
        assert params.num_views >= 3
        assert params.rms_reprojection_error >= 0.0

    def test_raises_on_insufficient_views(self, synth_cb_img) -> None:
        # Only one view -> calibration must fail.
        view = synth_cb_img((7, 6))
        with pytest.raises(ValueError):
            calibration.calibrate_from_images([view], (7, 6))

    def test_raises_on_inconsistent_shapes(self, synth_cb_img) -> None:
        a = synth_cb_img((7, 6))
        b = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2))
        with pytest.raises(ValueError):
            calibration.calibrate_from_images([a, b], (7, 6))

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError):
            calibration.calibrate_from_images([], (7, 6))


class TestUndistortFrame:
    r"""Unit tests for undistortion of frames using camera parameters."""

    def test_zero_distortion_is_identity(self) -> None:
        params = calibration.CameraParameters(
            camera_matrix=np.array(
                [[1000.0, 0.0, 480.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]
            ),
            dist_coeffs=np.zeros(5),
            img_size=(960, 720),
            rms_reprojection_error=0.0,
            pattern_size=(7, 6),
            square_size=0.025,
            num_views=10,
        )
        frame = (np.random.rand(720, 960, 3) * 255).astype(np.uint8)
        out = calibration.undistort_frame(frame, params)
        assert out.shape == frame.shape
        np.testing.assert_allclose(out, frame, atol=1)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
