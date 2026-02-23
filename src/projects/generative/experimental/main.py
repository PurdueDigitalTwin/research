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
flags.DEFINE_integer(
    name="batch_size",
    default=128,
    required=False,
    help="Per-device batch size for training.",
)
flags.DEFINE_integer(
    name="max_training_steps",
    default=400_000,
    required=False,
    help="Maximum number of training steps to run.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    required=False,
    help="Random seed for experiment reproducibility.",
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

    from src.projects.generative.experimental import run_ua_flow

    status = run_ua_flow.train(
        batch_size=int(flags.FLAGS.batch_size),
        max_training_steps=int(flags.FLAGS.max_training_steps),
        seed=int(flags.FLAGS.seed),
    )

    jax.distributed.shutdown()

    return status


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
