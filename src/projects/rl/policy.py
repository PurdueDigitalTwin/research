import typing

import flax.linen as nn
import jax


# Define the MLP policy network using Flax Linen (fully connected layers)
# this is a sckeleton architecture, we need another implementation for DQN model
# NOTE: sometimes the outputs are actions, but for DQN, the outputs are Q-values
# for all actions.
class MlpPolicy(nn.Module):
    r"""Multi-layer Perceptron Policy Network.

    Attributes:
        features (int): Dimensionality of the hidden features.
        out_features (int): Dimensionality of the output layer, typically 
            equal to the action space dimension.
        num_layers (int): Number of fully connected layers in the network.
        activation_fn (typing.Callable): Activation function to apply after each
            hidden layer.
        dtype (Optional[typing.Any], optional): Data type for the network
            computations. 
        param_dtype (Optional[typing.Any], optional): Data type for the network
            parameters.
    """

    features: int
    out_features: int
    num_layers: int
    activation_fn: typing.Callable
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
                # apply activation function for hidden layers
                out = self.activation_fn(out)

        return out
    

class ActorCriticPolicy(nn.Module):
    r"""Actor-Critic Policy Network.

    Attributes:
        features (int): Dimensionality of the hidden features.
        out_features (int): Dimensionality of the output layer, typically 
            equal to the action space dimension.
        num_layers (int): Number of fully connected layers in the network.
        activation_fn (typing.Callable): Activation function to apply after each
            hidden layer.
        dtype (Optional[typing.Any], optional): Data type for the network
            computations. 
        param_dtype (Optional[typing.Any], optional): Data type for the network
            parameters.
    """

    features: int
    out_features: int
    num_layers: int
    activation_fn: typing.Callable
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
        r"""Forward pass the policy network `\pi(a|s;\theta)`

        Args:
            inputs (jax.Array): Input state array of shape `(*, D)`

        Returns:
            For PPO: Action probabilities for all actions, with shape 
                `(*, out_features)`.
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
                # apply activation function for hidden layers
                out = self.activation_fn(out)

        # Multi-head actor-critic sharing the same backbone network.
        out = self.activation_fn(out)

        # Actor Head (logits for action probabilities)
        logits = nn.Dense(
            features=self.out_features,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="actor_head",
        )(out)

        # Critic Head (state value estimation)
        value = nn.Dense(
            features=1,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="critic_head",
        )(out)

        return logits, value
