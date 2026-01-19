import dataclasses
import typing

import fiddle as fdl
import optax

from src.core import datamodule as _datamodule
from src.core import model as _model


@dataclasses.dataclass(frozen=False, kw_only=True)
class DataConfig:
    r"""Configurations for data module.

    Attributes:
        module (fiddle.Partial): A factory function to create a `DataModule`.
        batch_size (int): The batch size used for data loading. In a multi-host
            multi-process environment, this is de batch size for each device.
        num_workers (int): The number of parallel workers for data loading.
        deterministic (bool): Whether to enforce deterministic data loading.
        drop_remainder (bool): Whether to drop the last incomplete batch.
    """

    module: fdl.Partial[_datamodule.DataModule]
    batch_size: int = 32
    num_workers: int = 4
    deterministic: bool = True
    drop_remainder: bool = False

    def __repr__(self) -> str:
        """Custom repr string for better readability."""
        fields = dataclasses.fields(self)
        field_string = []
        for field in fields:
            if field.name.startswith("_"):
                continue
            value = getattr(self, field.name)
            field_string.append(f"\t{field.name}={value!r}")
        field_string_joined = ",\n".join(field_string)
        return f"{self.__class__.__name__}(\n{field_string_joined}\n)"


@dataclasses.dataclass(frozen=False, kw_only=True)
class TrainerConfig:
    r"""Configuration for the training loop and evaluation.

    Attributes:
        num_train_steps (int): Total number of training steps.
        checkpoint_every_n_steps (Optional[int]): Frequency of checkpointing.
            If `None`, defaults to `eval_every_n_steps`.
        log_every_n_steps (int): Frequency of logging training metrics.
        eval_every_n_steps (int): Frequency of evaluation during training.
        checkpoint_dir (Optional[str]): Directory of checkpoint to resume from.
        max_checkpoints_to_keep (int): Maximum number of checkpoints to keep.
        profile (bool): Whether to enable profiling during training.
    """

    num_train_steps: int = 10_000
    checkpoint_every_n_steps: typing.Optional[int] = None
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 1_000
    checkpoint_dir: typing.Optional[str] = None
    max_checkpoints_to_keep: int = 1
    profile: bool = False

    def __repr__(self) -> str:
        """Custom repr string for better readability."""
        fields = dataclasses.fields(self)
        field_string = []
        for field in fields:
            if field.name.startswith("_"):
                continue
            value = getattr(self, field.name)
            field_string.append(f"\t{field.name}={value!r}")
        field_string_joined = ",\n".join(field_string)
        return f"{self.__class__.__name__}(\n{field_string_joined}\n)"


@dataclasses.dataclass(frozen=False, kw_only=True)
class OptimizerConfig:
    r"""Configuration for the optimizer and learning rate schedule."""

    lr_schedule: fdl.Config[typing.Callable]
    optimizer: fdl.Partial[optax.GradientTransformation]
    grad_clip_method: typing.Optional[typing.Literal["norm", "value"]] = None
    grad_clip_value: float = 1.0
    ema_rate: float = 0.0

    def __repr__(self) -> str:
        """Custom repr string for better readability."""
        fields = dataclasses.fields(self)
        field_string = []
        for field in fields:
            if field.name.startswith("_"):
                continue
            value = getattr(self, field.name)
            field_string.append(f"{field.name}={value!r}")
        field_string_joined = ",\n".join(field_string)
        return f"{self.__class__.__name__}(\n{field_string_joined}\n)"


@dataclasses.dataclass(frozen=False, kw_only=True)
class ExperimentConfig:
    r"""The master configuration for a complete experiment.

    Attributes:
        project_name (str): The name of the project.
        exp_name (str): The name of the experiment tag.
        mode (str): The running mode, one of "train", "evaluate", "inference".
        data (DataConfig): The data module configuration.
        trainer (TrainerConfig): The trainer configuration.
        optimizer (OptimizerConfig): The optimizer configuration.
        model (fiddle.Partial): A factory function to create the model.
        metric (Optional[fiddle.Config[Callable]]): A factory function to create
            the evaluation metric.
        dtype (Any): The global computation dtype.
        param_dtype (Any): The parameter dtype.
        precision (Any): The precision policy for computation.
        seed (int, optional): The random seed for initialization.
            Default is `42`.
    """

    project_name: str
    exp_name: str
    mode: typing.Literal["train", "evaluate", "inference"]

    # Composed configuration objects
    data: DataConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig

    # Fiddle config for the model, which implements the BaseModel interface
    model: fdl.Partial[_model.Model]

    # Fiddle config for the metric, which is a callable that outputs arrays
    metric: typing.Optional[fdl.Config[typing.Callable]] = None

    # Global settings
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Sanity checks."""
        assert isinstance(self.seed, int) and self.seed >= 0

    def __repr__(self) -> str:
        """Custom repr string for better readability."""
        fields = dataclasses.fields(self)
        field_string = []
        for field in fields:
            if field.name.startswith("_"):
                continue
            value = getattr(self, field.name)
            field_string.append(f"{field.name}={value!r}")
        field_string_joined = ",\n".join(field_string)
        return f"{self.__class__.__name__}(\n{field_string_joined}\n)"
