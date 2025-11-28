import typing

import chex
from flax import linen as nn
import jax


class ResNetBlock(nn.Module):
    r"""A residual downsampling block with two convolutional layers.

    Args:
        features (int): Dimensionality of the latent feaatures.
        num_groups (int, optional): Number of groups for `GroupNorm`.
            Default is :math:`32`.
        epsilon (float, optional): Small float added to variance to avoid
            dividing by zero in `GroupNorm`. Default is :math:`1e-5`.
        deterministic (bool, optional): If true, the model is run in
            deterministic mode (e.g., no dropout). Defaults to `None`.
        dropout_rate (float, optional): Dropout rate. Default is :math:`0`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    num_groups: int = 32
    epsilon: float = 1e-5
    deterministic: typing.Optional[bool] = None
    dropout_rate: float = 0.0
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    def setup(self) -> None:
        r"""Instantiates a `ResNetBlock` instance."""
        self.norm_1 = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm0",
        )
        self.conv_1 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_in",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv0",
        )
        self.cond_linear = nn.Dense(
            features=self.features,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_in",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="cond_in",
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate, name="dropout")

        self.norm_2 = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm1",
        )
        self.conv_2 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_in",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )
        self.conv_shortcut = nn.Dense(
            features=self.features,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_in",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_shortcut",
        )

    def __call__(
        self,
        inputs: jax.Array,
        cond: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the `ResNetBlock`.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C_in)`.
            cond (Optional[jax.Array], optional): Optional conditioning array
                of shape `(*, C_cond)`.
            deterministic (bool, optional): If true, the model is run in
                deterministic mode (e.g., no dropout). Defaults to `None`.

        Returns:
            Output array of shape `(*, H, W, C_out)`, where `C_out` is the
                `features` specified during instantiation.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        batch_dims = inputs.shape[:-3]
        dims = chex.Dimensions(
            H=inputs.shape[-3],
            W=inputs.shape[-2],
            C=inputs.shape[-1],
        )

        out = self.conv_1(nn.silu(self.norm_1(inputs)))

        if cond is not None:
            chex.assert_shape(cond, (*batch_dims, cond.shape[-1]))
            out = out + self.cond_linear(cond)[..., None, None, :]
        out = nn.silu(self.norm_2(out))
        out = self.dropout(out, deterministic=m_deterministic)
        out = self.conv_2(out)

        if inputs.shape[-1] != self.features:
            shortcut = self.conv_shortcut(inputs)
        else:
            shortcut = inputs
        out = out + shortcut
        chex.assert_shape(out, (*batch_dims, *dims["HW"], self.features))

        return out
