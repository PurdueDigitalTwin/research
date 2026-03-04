import jax
import jaxtyping


def shard(tree: jaxtyping.PyTree) -> jaxtyping.PyTree:
    r"""Helper function for ``jax.pmap`` to shard a pytree onto local devices.

    Args:
        tree (PyTree): The pytree to shard.

    Returns:
        A ``PyTree`` with an added leading dimension for local devices.
    """
    _shape_prefix = (jax.local_device_count(), -1)
    return jax.tree_util.tree_map(
        lambda x: (
            x.reshape(_shape_prefix + x.shape[1:])
            if hasattr(x, "reshape")
            else x
        ),
        tree=tree,
    )
