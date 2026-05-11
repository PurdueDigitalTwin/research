import os
import typing

from absl import app
from absl import flags

from src.utilities import logging

# Flags
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Directory for output files.",
)


def main(argv: typing.List[str]) -> int:
    r"""Main entry point for vehicle trajectory extraction."""
    del argv  # unused console kwargs

    FLAGS = flags.FLAGS
    logging.rank_zero_info("Working directory %s.", FLAGS.work_dir)

    return 0


if __name__ == "__main__":
    app.run(main=main)
