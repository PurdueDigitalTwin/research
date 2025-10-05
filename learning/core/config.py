import dataclasses
import typing

import fiddle as fdl

from learning.core import mixin as _mixin


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    """str: Name of the experiment."""

    # dataset configurations
    data: fdl.Partial[_mixin.DataMixin]
    """fdl.Partial: Partial configurations for building the dataset."""

    # model configurations
    model: fdl.Config
    """fdl.Config: Configurations for building the model."""

    # optimizer configurations
    lr_scheduler: fdl.Config
    """fdl.Config: Configurations for building the learning rate scheduler."""
    optimizer: fdl.Partial
    """fdl.Partial: Configurations for building the optimizer."""
    ema_rate: float = 0.0
    """float: Rate for exponential moving average of parameters."""

    # global training configurations
    batch_size: int = 32
    """int: Batch size for data loading."""
    deterministic: bool = True
    """int: Whether to use deterministic behavior in data loading and model."""
    drop_remainder: bool = False
    """bool: Whether to drop the last incomplete batch in an epoch."""
    num_workers: int = 4
    """int: Number of sharded workers for parallel data loading."""
    checkpoint_dir: str = "./logs/checkpoints"
    """str: Top-level directory for saving checkpoints."""
    num_train_steps: int = 10_000
    """int: Number of training steps."""
    check_val_every_n_steps: int = 1_000
    """int: Frequency for performing a validation loop."""
    grad_clip_method: typing.Optional[typing.Literal["norm", "value"]] = None
    """Optional[str]: Gradient clipping method, 'norm', 'value', or `None`."""
    grad_clip_value: float = 1.0
    """float: Gradient clipping value."""
    log_every_n_steps: int = 50
    """int: Frequency for logging to console and writing to loggers."""
    precision: typing.Literal["bfloat16", "float32"] = "float32"
    """str: Numerical precision for training. One of 'bfloat16' or 'float32'."""
    resume: bool = False
    """bool: Whether to resume from the latest checkpoint."""
    seed: int = 42
    """int: Random seed """
    train: bool = True
    """bool: Whether to run the training loop."""

    def __post_init__(self) -> None:
        """Post-initialization sanity checks."""
        assert isinstance(self.data, fdl.Partial), (
            "The `data` field must be of type `fdl.Partial`, "
            f"but got {type(self.data)} instead."
        )
        assert isinstance(self.model, fdl.Config), (
            "The `model` field must be of type `fdl.Config`, "
            f"but got {type(self.model)} instead."
        )
        assert isinstance(self.lr_scheduler, fdl.Config), (
            "The `lr_scheduler` field must be of type `fdl.Config`, "
            f"but got {type(self.lr_scheduler)} instead."
        )
        assert isinstance(self.optimizer, fdl.Partial), (
            "The `optimizer` field must be of type `fdl.Partial`, "
            f"but got {type(self.optimizer)} instead."
        )
        if self.ema_rate < 0.0 or self.ema_rate > 1.0:
            raise ValueError("The `ema_rate` must be in the range [0.0, 1.0].")
        if self.precision not in ("bfloat16", "float32"):
            raise ValueError(
                f"Unsupported precision {self.precision}. "
                "Supported precisions are 'bfloat16' and 'float32'."
            )

    def __repr__(self) -> str:
        """String representation of the class."""
        fields = dataclasses.fields(self)
        field_strings = []
        for field in fields:
            if field.name.startswith("_"):
                # skip the private fields
                continue
            value = getattr(self, field.name)
            field_strings.append(f"{field.name}={value!r}")
        field_strings_joined = ",\n".join(field_strings)
        return f"{self.__class__.__name__}(\n{field_strings_joined}\n)"
