import typing

from absl import app
from absl import flags
from fiddle import absl_flags
import jax

from src.core import distributed as _distributed

CONFIG = absl_flags.DEFINE_fiddle_config(
    name="experiment",
    default=None,
    help_string="Function to call for experiment configurations.",
    required=False,
)
flags.DEFINE_bool(
    name="distributed",
    default=False,
    help="Whether to enable multi-host distributed training in JAX.",
    required=False,
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    help="Directory to store experiment results.",
    required=True,
)


def main(_: typing.List[str]) -> None:
    r"""Main entry point for behavior models."""
    del _  # unused argument

    if flags.FLAGS.distributed:
        _distributed.setup_jax_distributed()

    from src.projects.generative import experiment

    experiment.train_and_evaluate(
        exp_config=CONFIG.value,
        work_dir=flags.FLAGS.work_dir,
    )
    jax.distributed.shutdown()


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
