import typing

import flax.linen as nn
import jax


# Define the MLP policy network using Flax Linen (fully connected layers)
# this is a sckeleton architecture, we need another implementation for DQN model
# NOTE: sometimes the outputs are actions, but for DQN, the outputs are Q-values for all actions.
class MlpPolicy(nn.Module):  
    r"""Multi-layer Perceptron Policy Network.

    Attributes:
        features (int): Dimensionality of the hidden features.
    """

    features: int
    out_features: int
    num_layers: int
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass the policy network `\pi(a|s;\theta)`

        Args:
            inputs (jax.Array): Input state array of shape `(*, D)`

        Returns:
            Raw Q-values for all actions, with shape `(*, out_features)`
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
                    mode="fan_avg",  # fan_avg means average of fan_in and fan_out, fan_in means input dim, fan_out means output dim
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
                out = jax.nn.relu(
                    out
                )  # apply relu activation function for hidden layers

        return out
