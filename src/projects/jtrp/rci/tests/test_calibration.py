import sys
import typing

import cv2
from numpy import typing as npt
import numpy as np
import pytest

from src.projects.jtrp.rci import calibration


@pytest.fixture(scope="module")
def _synth_cb_img(
    pattern_size: typing.Tuple[int, int] = (7, 6),
    square_px: int = 60,
    margin: int = 60,
    image_size: typing.Tuple[int, int] = (960, 720),
) -> npt.NDArray:
    r"""Renders a synthetic chessboard image on a white background.

    The chessboard has ``cols + 1`` and ``rows + 1`` total squares (so
    ``cols x rows`` interior corners), matching the OpenCV convention.
    """
    cols, rows = pattern_size
    canvas = np.full((image_size[1], image_size[0], 3), 220, dtype=np.uint8)
    total_cols, total_rows = cols + 1, rows + 1
    board_w = total_cols * square_px
    board_h = total_rows * square_px
    x0 = margin
    y0 = margin
    if x0 + board_w > image_size[0] or y0 + board_h > image_size[1]:
        raise ValueError("Chessboard does not fit in image_size with margins.")

    for r in range(total_rows):
        for c in range(total_cols):
            color = 0 if (r + c) % 2 == 0 else 255
            xa = x0 + c * square_px
            ya = y0 + r * square_px
            xb = xa + square_px
            yb = ya + square_px
            canvas[ya:yb, xa:xb] = color
    return canvas


class TestDetectChessboardCorners:
    r"""Unit tests for chessboard corner detection."""

    def test_detects_synthetic_pattern(self, _synth_cb_img) -> None:
        gray = cv2.cvtColor(_synth_cb_img, cv2.COLOR_BGR2GRAY)
        corners = calibration.detect_chessboard_corners(gray, (7, 6))
        assert corners is not None
        assert corners.shape == (7 * 6, 1, 2)

    def test_returns_none_for_non_chessboard(self) -> None:
        gray = np.full((480, 640), 128, dtype=np.uint8)
        corners = calibration.detect_chessboard_corners(gray, (7, 6))
        assert corners is None

    def test_accepts_color_input(self, _synth_cb_img) -> None:
        corners = calibration.detect_chessboard_corners(_synth_cb_img, (7, 6))
        assert corners is not None


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
