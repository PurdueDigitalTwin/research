import typing

import flax.linen as nn
import jax
from jax import numpy as jnp


# Define the MLP policy network using Flax Linen (fully connected layers)
# this is a sckeleton architecture, we need another implementation for DQN model
# NOTE: sometimes the outputs are actions, but for DQN, the outputs are Q-values
# for all actions.
class MlpPolicy(nn.Module):
    r"""Multi-layer Perceptron Policy Network.

    Attributes:
        features (int): Dimensionality of the hidden features.
        out_features (int): Dimensionality of the output space.
        num_layers (int): Number of layers in the MLP.
        activation (Callable[[jax.Array], jax.Array], optional): Activation
            function between layers. Default is ``jax.nn.relu``.
        dtype (Any, optional): Data type for the computations.
        param_dtype (Any, optional): Data type for the parameters.
    """

    features: int
    out_features: int
    num_layers: int
    activation: typing.Callable[[jax.Array], jax.Array] = jax.nn.relu
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass the policy network `\pi(a|s;\theta)`

        Args:
            inputs (jax.Array): Input state array of shape `(*, D)`

        Returns:
            For DQN: Raw Q-values for all actions, with shape `(*, out_features)`
        """
        out = inputs.astype(self.dtype)
        for i in range(self.num_layers):
            fc = nn.Dense(
                features=(
                    self.features
                    if i != self.num_layers - 1
                    else self.out_features
                ),
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",  # fan_avg means average of fan_in and fan_out
                    # fan_in means input dim, fan_out means output dim
                    distribution="uniform",  # uniform means uniform distribution
                ),
                use_bias=True,  # use bias term
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"fc_{i+1:d}",
            )
            out = fc(out)
            if i != self.num_layers - 1:
                out = self.activation(out)

        return out


class GaussianPolicy(nn.Module):
    r"""Gaussian Policy Network for continuous action spaces.

    Attributes:
        features (int): Dimensionality of the hidden features.
        out_features (int): Dimensionality of the action space.
        num_layers (int): Number of layers in the MLP.
        activation (Callable[[jax.Array], jax.Array]): Activation function
            to use between layers (e.g., ``jax.nn.relu``).
        use_state_dependent_std (bool, optional): Whether the standard deviation
            is state-dependent. Default is ``False``.
        dtype (Any, optional): Data type for the computations.
            Default is ``None``.
        param_dtype (Any, optional): Data type for the parameters.
            Default is ``None``.
    """

    features: int
    out_features: int
    num_layers: int
    activation: typing.Callable[[jax.Array], jax.Array]
    use_state_dependent_std: bool = False
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
    ) -> typing.Tuple[jax.Array, jax.Array]:
        r"""Forward pass the Gaussian policy network `\pi(a|s;\theta)`

        Args:
            inputs (jax.Array): Input state array of shape `(*, D)`

        Returns:
            A tuple of (mean, log_std) for the Gaussian policy, where both have shape `(*, out_features)`
        """
        out = inputs.astype(self.dtype)

        kernel_init = jax.nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_avg",  # fan_avg means average of fan_in and fan_out
            # fan_in means input dim, fan_out means output dim
            distribution="uniform",  # uniform means uniform distribution
        )

        for i in range(self.num_layers - 1):
            fc = nn.Dense(
                features=(
                    self.features
                    if i != self.num_layers - 1
                    else self.out_features
                    * 2  # output mean and log_std together
                ),
                kernel_init=kernel_init,
                use_bias=True,  # use bias term
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"fc_{i+1:d}",
            )
            out = fc(out)
            out = self.activation(out)

        if self.use_state_dependent_std:
            fc_out = nn.Dense(
                features=self.out_features
                * 2,  # output mean and log_std together
                kernel_init=kernel_init,
                use_bias=True,  # use bias term
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="fc_out",
            )
            out = fc_out(out)
            mean, log_std = jnp.split(
                out, 2, axis=-1
            )  # split into mean and log_std
            log_std = jnp.clip(
                log_std, -5.0, 2.0
            )  # clip log_std for numerical stability
        else:
            fc_mean = nn.Dense(
                features=self.out_features,
                kernel_init=kernel_init,
                use_bias=True,  # use bias term
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="fc_mean",
            )
            mean = fc_mean(out)

            log_std_param = self.param(
                "log_std",
                jax.nn.initializers.zeros,
                (self.out_features,),
                self.param_dtype,
            )
            log_std = jnp.broadcast_to(log_std_param, mean.shape)

        return mean, log_std
