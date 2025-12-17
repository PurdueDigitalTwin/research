import typing

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random as jrnd
from jax import typing as jax_typing

# ==============================================================================
# Coefficients of various wavelet decomposition low-pass filters.
WAVELETS = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db1": [0.7071067811865476, 0.7071067811865476],
    "db2": [
        -0.12940952255092145,
        0.22414386804185735,
        0.836516303737469,
        0.48296291314469025,
    ],
    "db3": [
        0.035226291882100656,
        -0.08544127388224149,
        -0.13501102001039084,
        0.4598775021193313,
        0.8068915093133388,
        0.3326705529509569,
    ],
    "db4": [
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ],
    "db5": [
        0.003335725285001549,
        -0.012580751999015526,
        -0.006241490213011705,
        0.07757149384006515,
        -0.03224486958502952,
        -0.24229488706619015,
        0.13842814590110342,
        0.7243085284385744,
        0.6038292697974729,
        0.160102397974125,
    ],
    "db6": [
        -0.00107730108499558,
        0.004777257511010651,
        0.0005538422009938016,
        -0.031582039318031156,
        0.02752286553001629,
        0.09750160558707936,
        -0.12976686756709563,
        -0.22626469396516913,
        0.3152503517092432,
        0.7511339080215775,
        0.4946238903983854,
        0.11154074335008017,
    ],
    "db7": [
        0.0003537138000010399,
        -0.0018016407039998328,
        0.00042957797300470274,
        0.012550998556013784,
        -0.01657454163101562,
        -0.03802993693503463,
        0.0806126091510659,
        0.07130921926705004,
        -0.22403618499416572,
        -0.14390600392910627,
        0.4697822874053586,
        0.7291320908465551,
        0.39653931948230575,
        0.07785205408506236,
    ],
    "db8": [
        -0.00011747678400228192,
        0.0006754494059985568,
        -0.0003917403729959771,
        -0.00487035299301066,
        0.008746094047015655,
        0.013981027917015516,
        -0.04408825393106472,
        -0.01736930100202211,
        0.128747426620186,
        0.00047248457399797254,
        -0.2840155429624281,
        -0.015829105256023893,
        0.5853546836548691,
        0.6756307362980128,
        0.3128715909144659,
        0.05441584224308161,
    ],
    "sym2": [
        -0.12940952255092145,
        0.22414386804185735,
        0.836516303737469,
        0.48296291314469025,
    ],
    "sym3": [
        0.035226291882100656,
        -0.08544127388224149,
        -0.13501102001039084,
        0.4598775021193313,
        0.8068915093133388,
        0.3326705529509569,
    ],
    "sym4": [
        -0.07576571478927333,
        -0.02963552764599851,
        0.49761866763201545,
        0.8037387518059161,
        0.29785779560527736,
        -0.09921954357684722,
        -0.012603967262037833,
        0.0322231006040427,
    ],
    "sym5": [
        0.027333068345077982,
        0.029519490925774643,
        -0.039134249302383094,
        0.1993975339773936,
        0.7234076904024206,
        0.6339789634582119,
        0.01660210576452232,
        -0.17532808990845047,
        -0.021101834024758855,
        0.019538882735286728,
    ],
    "sym6": [
        0.015404109327027373,
        0.0034907120842174702,
        -0.11799011114819057,
        -0.048311742585633,
        0.4910559419267466,
        0.787641141030194,
        0.3379294217276218,
        -0.07263752278646252,
        -0.021060292512300564,
        0.04472490177066578,
        0.0017677118642428036,
        -0.007800708325034148,
    ],
    "sym7": [
        0.002681814568257878,
        -0.0010473848886829163,
        -0.01263630340325193,
        0.03051551316596357,
        0.0678926935013727,
        -0.049552834937127255,
        0.017441255086855827,
        0.5361019170917628,
        0.767764317003164,
        0.2886296317515146,
        -0.14004724044296152,
        -0.10780823770381774,
        0.004010244871533663,
        0.010268176708511255,
    ],
    "sym8": [
        -0.0033824159510061256,
        -0.0005421323317911481,
        0.03169508781149298,
        0.007607487324917605,
        -0.1432942383508097,
        -0.061273359067658524,
        0.4813596512583722,
        0.7771857517005235,
        0.3644418948353314,
        -0.05194583810770904,
        -0.027219029917056003,
        0.049137179673607506,
        0.003808752013890615,
        -0.01495225833704823,
        -0.0003029205147213668,
        0.0018899503327594609,
    ],
}


# ----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.
def translate2d(
    tx: jax_typing.ArrayLike,
    ty: jax_typing.ArrayLike,
    **kwargs,
) -> jax.Array:
    r"""Construct a 2D translation matrix.

    Args:
        tx (ArrayLike): Translation along x-axis.
        ty (ArrayLike): Translation along y-axis.

    Returns:
        A three by three translation matrix for 2D transformations.
    """
    del kwargs
    tx, ty = jnp.asarray(tx), jnp.asarray(ty)
    batch_dims = jnp.shape(tx)
    ind_3 = jnp.eye(3, dtype=tx.dtype)
    if len(batch_dims) > 0:
        ind_3 = jnp.broadcast_to(ind_3, (*batch_dims, 3, 3))
    out = ind_3.at[..., 0, 2].set(tx)
    out = out.at[..., 1, 2].set(ty)
    return out


def translate3d(
    tx: jax_typing.ArrayLike,
    ty: jax_typing.ArrayLike,
    tz: jax_typing.ArrayLike,
    **kwargs,
) -> jax.Array:
    r"""Construct a 3D translation matrix.

    Args:
        tx (ArrayLike): Translation along x-axis.
        ty (ArrayLike): Translation along y-axis.
        tz (ArrayLike): Translation along z-axis.

    Returns:
        A four by four translation matrix for 3D transformations.
    """
    del kwargs
    tx, ty, tz = jnp.asarray(tx), jnp.asarray(ty), jnp.asarray(tz)
    batch_dims = jnp.shape(tx)
    ind_4 = jnp.eye(4, dtype=tx.dtype)
    if len(batch_dims) > 0:
        ind_4 = jnp.broadcast_to(ind_4, (*batch_dims, 4, 4))
    out = ind_4.at[..., :3, 3].set(jnp.stack([tx, ty, tz], axis=-1))
    return out


def scale2d(
    sx: jax_typing.ArrayLike,
    sy: jax_typing.ArrayLike,
    **kwargs,
) -> jax.Array:
    r"""Construct a 2D scaling matrix.

    Args:
        sx (ArrayLike): Scaling factor along x-axis.
        sy (ArrayLike): Scaling factor along y-axis.

    Returns:
        A three by three scaling matrix for 2D transformations.
    """
    del kwargs
    sx, sy = jnp.asarray(sx), jnp.asarray(sy)
    batch_dims = jnp.shape(sx)
    ind_3 = jnp.eye(3, dtype=sx.dtype)
    if len(batch_dims) > 0:
        ind_3 = jnp.broadcast_to(ind_3, (*batch_dims, 3, 3))
    out = ind_3.at[..., 0, 0].set(sx)
    out = out.at[..., 1, 1].set(sy)
    return out


def scale3d(
    sx: jax_typing.ArrayLike,
    sy: jax_typing.ArrayLike,
    sz: jax_typing.ArrayLike,
    **kwargs,
) -> jax.Array:
    r"""Construct a 3D scaling matrix.

    Args:
        sx (ArrayLike): Scaling factor along x-axis.
        sy (ArrayLike): Scaling factor along y-axis.
        sz (ArrayLike): Scaling factor along z-axis.

    Returns:
        A four by four scaling matrix for 3D transformations.
    """
    del kwargs
    sx, sy, sz = jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(sz)
    batch_dims = jnp.shape(sx)
    ind_4 = jnp.eye(4, dtype=sx.dtype)
    if len(batch_dims) > 0:
        ind_4 = jnp.broadcast_to(ind_4, (*batch_dims, 4, 4))
    out = ind_4.at[..., 0, 0].set(sx)
    out = out.at[..., 1, 1].set(sy)
    out = out.at[..., 2, 2].set(sz)
    return out


def rotate2d(theta: jax_typing.ArrayLike, **kwargs) -> jax.Array:
    r"""Construct a 2D rotation matrix.

    Args:
        theta (ArrayLike): Rotation angle in radians.

    Returns:
        A three by three rotation matrix for 2D transformations.
    """
    del kwargs
    theta = jnp.asarray(theta)
    cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
    batch_dims = jnp.shape(theta)
    ind_3 = jnp.eye(3, dtype=theta.dtype)
    if len(batch_dims) > 0:
        ind_3 = jnp.broadcast_to(ind_3, (*batch_dims, 3, 3))
    out = ind_3.at[..., 0:2, 0].set(jnp.stack([cos_theta, sin_theta], axis=-1))
    out = out.at[..., 0:2, 1].set(jnp.stack([-sin_theta, cos_theta], axis=-1))
    return out


def rotate3d(v: jax.Array, theta: jax_typing.ArrayLike, **kwargs) -> jax.Array:
    r"""Construct a 3D rotation matrix using axis-angle representation.

    Args:
        v (jax.Array): Rotation axis with a shape of `(..., 3)`.
        theta (ArrayLike): Rotation angle in radians.

    Returns:
        A four by four rotation matrix for 3D transformations.
    """
    del kwargs
    v, theta = jnp.asarray(v), jnp.asarray(theta)
    batch_dims = jnp.shape(theta)
    ind_4 = jnp.eye(4, dtype=theta.dtype)
    if len(batch_dims) > 0:
        ind_4 = jnp.broadcast_to(ind_4, (*batch_dims, 4, 4))
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
    out = ind_4.at[..., 0, 0].set(cos_theta + vx * vx * (1 - cos_theta))
    out = out.at[..., 0, 1].set(vx * vy * (1 - cos_theta) - vz * sin_theta)
    out = out.at[..., 0, 2].set(vx * vz * (1 - cos_theta) + vy * sin_theta)
    out = out.at[..., 1, 0].set(vy * vx * (1 - cos_theta) + vz * sin_theta)
    out = out.at[..., 1, 1].set(cos_theta + vy * vy * (1 - cos_theta))
    out = out.at[..., 1, 2].set(vy * vz * (1 - cos_theta) - vx * sin_theta)
    out = out.at[..., 2, 0].set(vz * vx * (1 - cos_theta) - vy * sin_theta)
    out = out.at[..., 2, 1].set(vz * vy * (1 - cos_theta) + vx * sin_theta)
    out = out.at[..., 2, 2].set(cos_theta + vz * vz * (1 - cos_theta))
    return out


def translate2d_inv(
    tx: jax_typing.ArrayLike,
    ty: jax_typing.ArrayLike,
    **kwargs,
) -> jax.Array:
    r"""Construct the inverse of a 2D translation matrix.

    Args:
        tx (ArrayLike): Original translation along x-axis.
        ty (ArrayLike): Original translation along y-axis.

    Returns:
        A three by three translation matrix to inverse 2D translation.
    """
    return translate2d(jnp.negative(tx), jnp.negative(ty), **kwargs)


def scale2d_inv(
    sx: jax_typing.ArrayLike,
    sy: jax_typing.ArrayLike,
    **kwargs,
) -> jax.Array:
    r"""Construct the inverse of a 2D scaling matrix.

    Args:
        sx (ArrayLike): Original scaling factor along x-axis.
        sy (ArrayLike): Original scaling factor along y-axis.

    Returns:
        A three by three scaling matrix to inverse 2D scaling.
    """
    return scale2d(1 / sx, 1 / sy, **kwargs)


def rotate2d_inv(
    theta: jax_typing.ArrayLike,
    **kwargs,
) -> jax.Array:
    r"""Construct the inverse of a 2D rotation matrix.

    Args:
        theta (ArrayLike): Original rotation angle in radians.

    Returns:
        A three by three rotation matrix to inverse 2D rotation.
    """
    return rotate2d(jnp.negative(theta), **kwargs)


# ==============================================================================
# Augmentation pipeline
class EDMAugmentor(nn.Module):
    r"""Image augmentation pipeline for generative models.

    Args:
        p (float): Base probability multiplier for all augmentations.
        xflip (float): Probability multiplier for flipping along the x-axis.
        yflip (float): Probability multiplier for flipping along the y-axis.
        rotate_int (float): Probability multiplier for integer rotation.
        translate_int (float): Probability multiplier for integer translation.
        translate_int_max (float): Range of integer translation as a factor
            relative to image dimensions.
        scale (float): Probability multiplier for isotropic scaling.
        rotate_frac (float): Probability multiplier for fractional rotation.
        aniso (float): Probability multiplier for anisotropic scaling.
        translate_frac (float): Probability multiplier for fractiona
            translation.
        scale_std (float): Log2 standard deviation of isotropic scaling.
        rotate_frac_max (float): Range of fractional rotation, where `1` equals
            to a full circle.
        aniso_std (float): Log2 standard deviation of anisotropic scaling.
        aniso_rotate_prob (float): Probability of doing anisotropic scaling
            with respect to a rotated coordinate frame.
        translate_frac_std (float): Standard deviation of fractional
            translation, relative to image dimensions.
        brightness (float): Probability multiplier for brightness adjustment.
        contrast (float): Probability multiplier for contrast adjustment.
        lumaflip (float): Probability multiplier for luma flipping.
        hue (float): Probability multiplier for hue rotation.
        saturation (float): Probability multiplier for saturation adjustment.
        brightness_std (float): Standard deviation of brightness adjustment.
        contrast_std (float): Log2 standard deviation of contrast adjustment.
        hue_max (float): Range of hue rotation, where `1` is a full circle.
        saturation_std (float): Log2 standard deviation of saturation.
    """

    p: float = 1

    # arguments for pixel blitting
    xflip: float = 0
    yflip: float = 0
    rotate_int: float = 0
    translate_int: float = 0
    translate_int_max: float = 0.125

    # arguments for geometric transformations
    scale: float = 0
    rotate_frac: float = 0
    aniso: float = 0
    translate_frac: float = 0
    scale_std: float = 0.2
    rotate_frac_max: float = 1
    aniso_std: float = 0.2
    aniso_rotate_prob: float = 0.5
    translate_frac_std: float = 0.125

    # arguments for color transformations
    brightness: float = 0
    contrast: float = 0
    lumaflip: float = 0
    hue: float = 0
    saturation: float = 0
    brightness_std: float = 0.2
    contrast_std: float = 0.5
    hue_max: float = 1
    saturation_std: float = 1

    @nn.compact
    def __call__(
        self,
        images: jax.Array,
    ) -> typing.Tuple[jax.Array, jax.Array]:
        r"""Apply augmentations to the input images.

        Args:
            images (jax.Array): Input images with a shape of `(..., H, W, C)`.
                Note that images should be floating-point array with values in the range `[0, 1]`.

        Returns:
            A tuple of the augmented images and the conditioning label vector.
        """
        batch_dims = images.shape[:-3]
        height, width, channels = images.shape[-3:]
        images = jnp.reshape(images, [-1, height, width, channels])
        num = images.shape[0]
        labels = [jnp.zeros([*batch_dims, 0])]

        # ==============================================
        # Pixel blitting.
        key = self.make_rng("augment")

        if self.xflip > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jrnd.randint(w_key, [num, 1, 1, 1], 0, 2)
            w = jnp.where(
                jnp.less(
                    jrnd.uniform(u_key, shape=(num, 1, 1, 1)),
                    self.xflip * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            images = jnp.where(w == 1, jnp.flip(images, axis=-2), images)
            labels += [w]

        if self.yflip > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jrnd.randint(w_key, [num, 1, 1, 1], 0, 2)
            w = jnp.where(
                jnp.less(
                    jrnd.uniform(u_key, shape=(num, 1, 1, 1)),
                    self.yflip * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            images = jnp.where(w == 1, jnp.flip(images, axis=-3), images)
            labels += [w]

        if self.rotate_int > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jrnd.randint(w_key, [num, 1, 1, 1], 0, 4)
            w = jnp.where(
                jnp.less(
                    jrnd.uniform(u_key, shape=(num, 1, 1, 1)),
                    self.rotate_int * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            images = jnp.where(
                jnp.logical_or(w == 1, w == 2),
                jnp.flip(images, axis=-2),
                images,
            )
            images = jnp.where(
                jnp.logical_or(w == 2, w == 3),
                jnp.flip(images, axis=-3),
                images,
            )
            images = jnp.where(
                jnp.logical_or(w == 1, w == 3),
                jnp.swapaxes(images, -2, -3),
                images,
            )
            labels += [(w == 1) | (w == 2), (w == 2) | (w == 3)]

        if self.translate_int > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jrnd.uniform(w_key, [2, num, 1, 1, 1]) * 2 - 1
            w = jnp.where(
                jnp.less(
                    jrnd.uniform(u_key, shape=(1, num, 1, 1, 1)),
                    self.translate_int * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            tx = jnp.astype(
                jnp.round(w[0] * (width * self.translate_int_max)),
                jnp.int32,
            )
            ty = jnp.astype(
                jnp.round(w[1] * (height * self.translate_int_max)),
                jnp.int32,
            )

            b_idx = jnp.arange(num)[:, None, None, None]
            y_idx, x_idx = jnp.meshgrid(
                jnp.arange(height),
                jnp.arange(width),
                indexing="ij",
            )
            x_new = jnp.abs(
                width
                - 1
                - (width - 1 - (x_idx[None, ...] - tx) % (width * 2 - 2))
            )
            y_new = jnp.abs(
                height
                - 1
                - (height - 1 - (y_idx[None, ...] + ty) % (height * 2 - 2))
            )
            images = images[b_idx, y_new, x_new, :]
            labels += [
                jnp.divide(tx, (width * self.translate_int_max)),
                jnp.divide(ty, (height * self.translate_int_max)),
            ]

        # =============================================
        # geometric transformations.
        # ind_3 = jnp.eye(3)
        # g_inv = ind_3

        # if self.scale > 0:
        #     key, w_key, u_key = jrnd.split(key, 3)
        #     w = jrnd.normal(w_key, [num])
        #     w = jnp.where(
        #         jnp.less(
        #             jrnd.uniform(u_key, shape=(num,)),
        #             self.scale * self.p,
        #         ),
        #         w,
        #         jnp.zeros_like(w),
        #     )
        #     # s = w.mul(self.scale_std).exp2()
        #     s = 2 ** (w * self.scale_std)
        #     g_inv = g_inv @ scale2d_inv(s, s)
        #     labels += [w]

        # if self.rotate_frac > 0:
        #     key, w_key, u_key = jrnd.split(key, 3)
        #     w = jnp.multiply(
        #         jrnd.uniform(w_key, [num]) * 2 - 1,
        #         np.pi * self.rotate_frac_max,
        #     )
        #     w = jnp.where(
        #         jnp.less(
        #             jrnd.uniform(u_key, shape=(num,)),
        #             self.rotate_frac * self.p,
        #         ),
        #         w,
        #         jnp.zeros_like(w),
        #     )
        #     g_inv = g_inv @ rotate2d_inv(-w)
        #     labels += [jnp.cos(w) - 1, jnp.sin(w)]

        # if self.aniso > 0:
        #     key, w_key, r_key, u_key1, u_key2 = jrnd.split(key, 5)
        #     w = jrnd.normal(w_key, [num])
        #     r = jnp.multiply(jrnd.uniform(r_key, [num]) * 2 - 1, jnp.pi)
        #     w = jnp.where(
        #         jnp.less(
        #             jrnd.uniform(u_key1, shape=(num,)),
        #             self.aniso * self.p,
        #         ),
        #         w,
        #         jnp.zeros_like(w),
        #     )
        #     r = jnp.where(
        #         jnp.less(
        #             jrnd.uniform(u_key2, shape=(num,)),
        #             self.aniso_rotate_prob,
        #         ),
        #         r,
        #         jnp.zeros_like(r),
        #     )
        #     # s = w.mul(self.aniso_std).exp2()
        #     s = 2 ** (jnp.multiply(w, self.aniso_std))
        #     g_inv = (
        #         g_inv
        #         @ rotate2d_inv(r)
        #         @ scale2d_inv(s, 1 / s)
        #         @ rotate2d_inv(-r)
        #     )
        #     labels += [w * jnp.cos(r), w * jnp.sin(r)]

        # if self.translate_frac > 0:
        #     key, w_key, u_key = jrnd.split(key, 3)
        #     w = jrnd.normal(w_key, [2, num])
        #     w = jnp.where(
        #         jnp.less(
        #             jrnd.uniform(u_key, shape=(1, num)),
        #             self.translate_frac * self.p,
        #         ),
        #         w,
        #         jnp.zeros_like(w),
        #     )
        #     g_inv = g_inv @ translate2d_inv(
        #         jnp.multiply(w[0], (width * self.translate_frac_std)),
        #         jnp.multiply(w[1], (height * self.translate_frac_std)),
        #     )
        #     labels += [w[0], w[1]]

        # if g_inv is not ind_3:
        #     cx = (width - 1) / 2
        #     cy = (height - 1) / 2
        #     cp = matrix(
        #         [-cx, -cy, 1],
        #         [cx, -cy, 1],
        #         [cx, cy, 1],
        #         [-cx, cy, 1],
        #     )  # [idx, xyz]
        #     cp = g_inv @ jnp.matrix_transpose(cp)  # [batch, xyz, idx]
        #     Hz = jnp.asarray(WAVELETS["sym6"], dtype=jnp.float32)
        #     Hz_pad = len(Hz) // 4
        #     margin = jnp.transpose(cp[:, :2, :], axes=(1, 0, 2))
        #     margin = jnp.reshape(margin, [2, -1])  # [x/y, corners]
        #     margin = jnp.max(
        #         jnp.concatenate([-margin, margin]),
        #         axis=1,
        #     )  # [x0, y0, x1, y1]
        #     margin = margin + jnp.array([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2)
        #     margin = margin.max(jnp.array([0, 0] * 2))
        #     margin = margin.min(constant([W - 1, H - 1] * 2, device=device))
        #     mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

        #     # Pad image and adjust origin.
        #     images = jnp.pad(
        #         images,
        #         pad_width=[mx0, mx1, my0, my1],
        #         mode="reflect",
        #     )
        #     g_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ g_inv

        #     # Upsample.
        #     conv_weight = jnp.tile(
        #         jnp.array(
        #             Hz[None, None, ::-1],
        #             dtype=images.dtype,
        #             device=images.device,
        #         ),
        #         [images.shape[1], 1, 1],
        #     )
        #     conv_pad = (len(Hz) + 1) // 2
        #     images = jnp.stack(
        #         [images, jnp.zeros_like(images)], axis=4
        #     ).reshape(num, channels, images.shape[2], -1)[:, :, :, :-1]
        #     images = torch.nn.functional.conv2d(
        #         images,
        #         conv_weight.unsqueeze(2),
        #         groups=images.shape[1],
        #         padding=[0, conv_pad],
        #     )
        #     images = torch.stack(
        #         [images, torch.zeros_like(images)], dim=3
        #     ).reshape(N, C, -1, images.shape[3])[:, :, :-1, :]
        #     images = torch.nn.functional.conv2d(
        #         images,
        #         conv_weight.unsqueeze(3),
        #         groups=images.shape[1],
        #         padding=[conv_pad, 0],
        #     )
        #     G_inv = (
        #         scale2d(2, 2, device=device)
        #         @ G_inv
        #         @ scale2d_inv(2, 2, device=device)
        #     )
        #     G_inv = (
        #         translate2d(-0.5, -0.5, device=device)
        #         @ G_inv
        #         @ translate2d_inv(-0.5, -0.5, device=device)
        #     )

        #     # Execute transformation.
        #     shape = [N, C, (H + Hz_pad * 2) * 2, (W + Hz_pad * 2) * 2]
        #     G_inv = (
        #         scale2d(
        #             2 / images.shape[3], 2 / images.shape[2], device=device
        #         )
        #         @ G_inv
        #         @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
        #     )
        #     grid = torch.nn.functional.affine_grid(
        #         theta=G_inv[:, :2, :], size=shape, align_corners=False
        #     )
        #     images = torch.nn.functional.grid_sample(
        #         images,
        #         grid,
        #         mode="bilinear",
        #         padding_mode="zeros",
        #         align_corners=False,
        #     )

        #     # Downsample and crop.
        #     conv_weight = constant(
        #         Hz[None, None, :], dtype=images.dtype, device=images.device
        #     ).tile([images.shape[1], 1, 1])
        #     conv_pad = (len(Hz) - 1) // 2
        #     images = torch.nn.functional.conv2d(
        #         images,
        #         conv_weight.unsqueeze(2),
        #         groups=images.shape[1],
        #         stride=[1, 2],
        #         padding=[0, conv_pad],
        #     )[:, :, :, Hz_pad:-Hz_pad]
        #     images = torch.nn.functional.conv2d(
        #         images,
        #         conv_weight.unsqueeze(3),
        #         groups=images.shape[1],
        #         stride=[2, 1],
        #         padding=[conv_pad, 0],
        #     )[:, :, Hz_pad:-Hz_pad, :]

        # ============================================
        # color transformations.
        ind_4 = jnp.eye(4)
        mat = ind_4
        luma_axis = jnp.true_divide(jnp.asarray([1, 1, 1, 0]), jnp.sqrt(3))

        if self.brightness > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jax.random.uniform(w_key, shape=(num,))
            w = jnp.where(
                jnp.less(
                    jax.random.uniform(u_key, shape=(num,)),
                    self.brightness * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            b = w * self.brightness_std
            mat = translate3d(b, b, b) @ mat
            labels += [w]

        if self.contrast > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jax.random.normal(w_key, shape=(num,))
            w = jnp.where(
                jnp.less(
                    jax.random.uniform(u_key, shape=(num,)),
                    self.contrast * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            c = jnp.exp2(jnp.multiply(w, self.contrast_std))
            mat = scale3d(c, c, c) @ mat
            labels += [w]

        if self.lumaflip > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jax.random.randint(w_key, (num, 1, 1), 0, 2)
            w = jnp.where(
                jnp.less(
                    jax.random.uniform(u_key, shape=(num, 1, 1)),
                    self.lumaflip * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            luma_outer = jnp.outer(luma_axis, luma_axis)
            correction = 2 * luma_outer[None, ...] * w
            mat = (ind_4[None, ...] - correction) @ mat
            labels += [w]

        if self.hue > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jax.random.uniform(w_key, shape=(num,)) * 2 - 1
            w = w * (jnp.pi * self.hue_max)
            w = jnp.where(
                jnp.less(
                    jax.random.uniform(u_key, shape=(num,)),
                    self.hue * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            luma_vec = jnp.tile(luma_axis[None, :3], (num, 1))
            mat = rotate3d(luma_vec, w) @ mat
            labels += [jnp.cos(w) - 1, jnp.sin(w)]

        if self.saturation > 0:
            key, w_key, u_key = jrnd.split(key, 3)
            w = jax.random.normal(w_key, shape=(num, 1, 1))
            w = jnp.where(
                jnp.less(
                    jax.random.uniform(u_key, shape=(num, 1, 1)),
                    self.saturation * self.p,
                ),
                w,
                jnp.zeros_like(w),
            )
            luma_outer = jnp.outer(luma_axis, luma_axis)
            s = jnp.exp2(w * self.saturation_std)
            mat = (
                luma_outer[None, ...]
                + (ind_4[None, ...] - luma_outer[None, ...]) * s
            ) @ mat
            labels += [w]

        images = images.reshape([num, height * width, channels])
        if channels == 3:
            images = images @ mat[:, :3, :3]
            images = images + jnp.matrix_transpose(mat[:, :3, 3:])
        elif channels == 1:
            mat = jnp.mean(mat[:, :3, :], axis=-2, keepdims=True)
            images = images * jnp.sum(mat[:, :, :3], axis=-1, keepdims=True)
            images = images + jnp.matrix_transpose(mat[:, :, 3:])
        else:
            raise ValueError(
                "Image must be RGB (3 channels) or Grayscale (1 channel)"
            )
        images = images.reshape([num, height, width, channels])
        images = jnp.clip(images, 0.0, 1.0)

        # Post-processing non-leaky conditioning vector
        labels = jnp.concatenate(
            [jnp.astype(x, jnp.float32).reshape(num, -1) for x in labels],
            axis=-1,
        )
        images = jnp.reshape(images, [*batch_dims, height, width, channels])
        labels = jnp.reshape(labels, [*batch_dims, labels.shape[-1]])

        return images, labels
