import copy
import typing

from flax import struct
import jax
import jaxtyping
import optax


class TrainState(struct.PyTreeNode):
    r"""Train state with exponential moving average of parameters.

    Attributes:
        step (int): Counter incremented by calls to `apply_gradients()`.
        params (Dict): Model parameters to be updated by the optimizer.
        mutables (Optional[Dict]): Mutable variables used during network calls.
        tx (GradientTransformation): The `optax` optimizer.
        opt_state (OptState): The state of the `optax` optimizer.
        ema_params (Dict): Exponential moving average of the parameters.
        ema_rate (float, optional): Decay rate for exponential moving average.
        ema_update_period (int, optional): Update EMA every N steps.
            When set to 1, EMA is updated every step (default behavior).
    """

    step: int
    params: typing.Dict = struct.field(pytree_node=True)
    mutables: typing.Optional[typing.Dict] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    ema_params: typing.Dict = struct.field(pytree_node=True)
    ema_rate: float = 0.999
    ema_update_period: int = 1

    def apply_gradients(
        self,
        *,
        grads: jaxtyping.PyTree,
        **kwargs,
    ) -> "TrainState":
        r"""Applies a single gradient step and update the parameters.

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

        # Update EMA only every `ema_update_period` steps.
        should_update_ema = (self.step + 1) % self.ema_update_period == 0
        ema_decay = self.ema_rate**self.ema_update_period
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, p: jax.lax.cond(
                should_update_ema,
                lambda: ema + (1.0 - ema_decay) * (p - ema),
                lambda: ema,
            ),
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
        ema_update_period: int = 1,
        mutables: typing.Optional[jaxtyping.PyTree] = None,
        **kwargs,
    ) -> "TrainState":
        r"""Creates a new `TrainState` instance.

        Args:
            params (PyTree): Initial model parameters.
            tx (GradientTransformation): The `optax` optimizer.
            ema_rate (float, optional): Decay rate for exponential moving
                average of the model parameters. Default is :math:`0.999`.
            ema_update_period (int, optional): Update EMA every N steps.
                Default is ``1`` (every step).
            mutables (Optional[PyTree], optional): Additional mutable variables
                other than network parameters. Default is ``None``.

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
            ema_update_period=ema_update_period,
            mutables=mutables,
            **kwargs,
        )


class MultiTrainState(struct.PyTreeNode):
    r"""A collection of multiple train states.

    Attributes:
        step (int): Counter incremented by calls to `apply_gradients()`.
        substates (Dict[str, TrainState]): A dictionary of train states.
    """

    step: int
    substates: typing.Dict[str, TrainState]

    def apply_gradients(
        self,
        *,
        grads: typing.Dict[str, jaxtyping.PyTree],
        **kwargs,
    ) -> "MultiTrainState":
        new_step, new_substates = self.step + 1, {}
        for key, grad in grads.items():
            if key not in self.substates:
                raise ValueError(f"Invalid state {key}.")
            new_substates[key] = self.substates[key].apply_gradients(
                grads=grad,
                kwargs=kwargs,
            )
            if new_substates[key].step != new_step:
                raise ValueError(
                    "Inconsistent train step: "
                    f"{new_substates[key].step:d} and {new_step:d}."
                )

        return self.replace(step=self.step + 1, substates=new_substates)

    @classmethod
    def create(
        cls: typing.Type["MultiTrainState"],
        *,
        substates: typing.Dict[str, TrainState],
        **kwargs,
    ) -> "MultiTrainState":
        r"""Creates a new `MultiTrainState` instance.

        Args:
            substates (Dict[str, TrainState]): A dictionary of train states.

        Returns:
            A new `MultiTrainState` initialized with the given train states.
        """

        return cls(step=0, substates=substates, **kwargs)


__all__ = ["TrainState", "MultiTrainState"]
