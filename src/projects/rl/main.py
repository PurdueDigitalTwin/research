import typing

from absl import app
from absl import flags
from fiddle import absl_flags
import jax

from src.core import distributed as _distributed

# Flags
CONFIG = absl_flags.DEFINE_fiddle_config(
    name="experiment",
    default=None,
    help_string="Callable that returns an experiment configuration.",
    required=False,
)
flags.DEFINE_boolean(
    name="distributed",
    default=False,
    help="Whether to enable JAX multi-host multi-process distributed training.",
    required=False,
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    help="Directory to store experiment outputs.",
    required=True,
)


def main(argv: typing.List[str]) -> int:
    r"""Main entry point for reinforcement learning."""
    del argv  # unused command line arguments

    if flags.FLAGS.distributed:
        _distributed.setup_jax_distributed()

    if flags.FLAGS.distributed:
        jax.distributed.shutdown()

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
