import copy
import typing

from flax import struct
from flax.core import frozen_dict
import jax
import jaxtyping
import optax


class TrainState(struct.PyTreeNode):
    """Train state with exponential moving average of parameters.

    Attributes:
        step (int): Counter incremented by calls to `apply_gradients()`.
        params (FrozenDict): Model parameters to be updated by the optimizer.
        tx (GradientTransformation): The `optax` optimizer.
        opt_state (OptState): The state of the `optax` optimizer.
    """

    step: int
    """int: Counter incremented by calls to `apply_gradients()`."""
    params: frozen_dict.FrozenDict = struct.field(pytree_node=True)
    """FrozenDict: Model parameters to be updated by the optimizer."""
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    """optax.GradientTransformation: The `optax` optimizer."""
    opt_state: optax.OptState = struct.field(pytree_node=True)
    """OptState: The state of the `optax` optimizer."""
    ema_params: frozen_dict.FrozenDict = struct.field(pytree_node=True)
    """FrozenDict: Exponential moving average of parameters."""
    ema_rate: float = 0.999
    """float: Decay rate for exponential moving average of parameters."""

    def apply_gradients(
        self,
        *,
        grads: jaxtyping.PyTree,
        **kwargs,
    ) -> "TrainState":
        """Applies a single gradient step and update the parameters.

        Args:
            grads (PyTree): Gradients with the same structure as `.params`.
            kwargs: Additional dataclass attributes to be `.replace()`-ed.

        Returns:
            A new state with updated parameters and optimizer state.
        """
        updates, new_opt_state = self.tx.update(
            updates=grads,
            state=self.opt_state,
            params=self.params,
        )
        new_params = optax.apply_updates(params=self.params, updates=updates)
        new_ema_params = jax.tree_util.tree_map(
            lambda x, y: x + (1.0 - self.ema_rate) * (y - x),
            self.ema_params,
            new_params,
        )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls: typing.Type["TrainState"],
        *,
        params: jaxtyping.PyTree,
        tx: optax.GradientTransformation,
        ema_rate: float = 0.999,
        **kwargs,
    ) -> "TrainState":
        """Creates a new `TrainState` instance.

        Args:
            params (PyTree): Initial model parameters.
            tx (GradientTransformation): The `optax` optimizer.

        Returns:
            A new `TrainState` initialized with the given params and optimizer.
        """
        opt_state = tx.init(params)
        return cls(
            step=0,
            params=params,
            tx=tx,
            opt_state=opt_state,
            ema_params=copy.deepcopy(params),
            ema_rate=ema_rate,
            **kwargs,
        )
