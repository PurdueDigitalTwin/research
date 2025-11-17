import os
import platform
import typing

from absl import app
from absl import flags
import clu.platform
from fiddle import absl_flags as fdl_flags
import fiddle as fdl
import flax
import jax
from learning.core import config as _config
from learning.core import train as _train
from learning.utilities import logging
import tensorflow as tf

# Constants
_CONFIG = fdl_flags.DEFINE_fiddle_config(
    name="experiment",
    default=None,
    help_string="Function to generate base experiment configuration.",
    required=True,
)
_FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="work_dir",
    default=None,
    help="Directory to store the experiment outputs.",
    required=True,
)


def main(argv: typing.List[str]) -> int:
    """Main entry point for training and evaluation."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.rank_zero_info("Running on host: %s", platform.node())

    # disable GPU/TPU for TensorFlow
    tf.config.set_visible_devices([], "GPU")
    tf.config.set_visible_devices([], "TPU")

    # initialize JAX runtime
    logging.rank_zero_info(
        "Running with JAX backend: %s",
        jax.default_backend(),
    )
    logging.rank_zero_info(
        "Running on JAX process %d / %d",
        jax.process_index() + 1,
        jax.process_count(),
    )
    logging.rank_zero_info("Running with JAX devices: %r", jax.devices())

    # setup experiment info
    clu.platform.work_unit().set_task_status(
        "process index: %d, process count: %d"
        % (jax.process_index() + 1, jax.process_count())
    )
    clu.platform.work_unit().create_artifact(
        artifact_type=clu.platform.ArtifactType.DIRECTORY,
        artifact=os.path.abspath(_FLAGS.work_dir),
        description="Working directory.",
    )

    experiment_cfg: fdl.Buildable = _CONFIG.value
    assert isinstance(experiment_cfg, _config.ExperimentConfig), (
        "Expect `experiment` flag to be of type ExperimentConfig, "
        "but got %s." % type(experiment_cfg)
    )
    logging.rank_zero_info("Config:\n%s", experiment_cfg)

    if experiment_cfg.train:
        _train.train_and_evaluate(
            config=experiment_cfg,
            work_dir=_FLAGS.work_dir,
        )
    else:
        raise NotImplementedError("Inference is not implemented yet.")

    return 0


if __name__ == "__main__":
    flax.config.config_with_absl()
    app.run(main=main)
