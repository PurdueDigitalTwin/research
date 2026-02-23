import typing

from absl import app
from absl import flags
import jax

from src.core import distributed as _distributed

flags.DEFINE_boolean(
    name="distributed",
    default=False,
    help="Whether to enable multi-host distributed training in JAX.",
    required=False,
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    help="Directory to store the experiment outputs",
    required=True,
)


def main(_: typing.List[str]) -> int:
    r"""Main entry point for running experimental modules."""
    del _  # unused command line arguments

    if flags.FLAGS.distributed:
        _distributed.setup_jax_distributed()

    jax.distributed.shutdown()

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
