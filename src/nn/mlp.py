import typing

from flax import linen as nn
import jax


class MultiLayerPerceptron(nn.Module):
    r"""Multi-layer Perceptron (MLP).

    Args:
        feature_list (Sequence[int]): A list of latent and output feature sizes
            such that ``len(feature_list)`` is the number of layers in MLP.
        dropout_rate (float | Sequence[float], optional): Dropout probability
            of each hidden layer. If a sequence is given, it sets the dropout
            probability for each layer. Default is :math:`0.0`.
        use_bias (bool, optional): Whether to use bias in the linear layers.
            Default is ``True``.
        activation (Callable, optional): Non-linear activation function.
            Default is ``jax.nn.relu``.
        plain_last (bool, optional): If set to ``False``, will apply
            non-linearity and dropout to the last layer. Default is ``True``.
        deterministic (bool | None, optional): Whether to apply the dropout.
            Default is ``None``.
        dtype (Any, optional): The data type of the computation.
            Default is ``None``.
        param_dtype (Any, optional): The data type of the parameters.
            Default is ``None``.
        precision (Any, optional): Numerical precision of the computation.
            Default is ``None``.
    """

    feature_list: typing.Sequence[int]
    dropout_rate: typing.Union[float, typing.Sequence[float]] = 0.0
    use_bias: bool = True
    activation: typing.Callable[..., jax.Array] = jax.nn.relu
    plain_last: bool = True
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass the MLP module.

        Args:
            inputs (jax.Array): Input array with a shape of ``(*, D)``.
            deterministic (bool, optional): Whether to apply dropout.
                It merges with the module-level attribute `deterministic`.
                Default is `None`.

        Returns:
            Output array with a shape of ``(*, feature_list[-1])``.
        """
        if isinstance(self.dropout_rate, float):
            dropout_rate = [self.dropout_rate] * self.num_layers
        elif isinstance(self.dropout_rate, typing.Sequence):
            dropout_rate = self.dropout_rate
        else:
            raise TypeError(
                "`dropout_rate` should be either a float or "
                f"a sequence of floats, but got {type(self.dropout_rate)}"
            )
        assert len(dropout_rate) == self.num_layers

        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        out = inputs.astype(self.dtype)

        for i, features in enumerate(self.feature_list):
            out = nn.Dense(
                features=features,
                use_bias=self.use_bias,
                # TODO (juanwu): kernel and bias initialization
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=f"linear_{i+1:d}",
            )(out)
            if i != self.num_layers - 1 or (not self.plain_last):
                out = self.activation(out)
                out = nn.Dropout(rate=dropout_rate[i])(out, m_deterministic)

        return out

    @property
    def num_layers(self) -> int:
        r"""int: Number of layers."""
        return len(self.feature_list)
