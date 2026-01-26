import functools
import os
import typing

import chex
import datasets
from flax import serialization
from huggingface_hub import hf_hub_download
import jax
from jax import numpy as jnp
from jax import typing as jxt
import jaxtyping
from numpy import typing as npt
import numpy as np
from PIL import Image
from scipy import linalg as splin
from tqdm import auto as tqdm
from tqdm.contrib import logging as tqdm_logging

from src.projects.generative.model import inception
from src.utilities import logging


def _frechet_distance(
    mu_left: jxt.ArrayLike,
    cov_left: jxt.ArrayLike,
    mu_right: jxt.ArrayLike,
    cov_right: jxt.ArrayLike,
    eps: float = 0.000001,
) -> npt.NDArray[np.float_]:
    r"""Computes the Fréchet Distance between two multivariate Gaussians.

    Args:
        mu_left (ArrayLike): Mean vector of the first Gaussian of shape `(D,)`.
        cov_left (ArrayLike): Covariance matrix of the first Gaussian with
            a shape of `(D, D)`.
        mu_right (ArrayLike): Mean of the second Gaussian of shape `(D,)`.
        cov_right (jax.Array): Covariance matrix of the second Gaussian with
            a shape of `(D, D)`.
        eps (float, optional): Small value to add to the diagonal for numerical
            stability. Default is `0.000001`.

    Returns:
        The Fréchet Distance between the two Gaussians as a scalar array.
    """
    # sanity checks
    chex.assert_equal_shape([mu_left, mu_right])
    chex.assert_equal_shape([cov_left, cov_right])

    mu_left = np.atleast_1d(np.array(mu_left))
    mu_right = np.atleast_1d(np.array(mu_right))
    cov_left = np.atleast_2d(np.array(cov_left))
    cov_right = np.atleast_2d(np.array(cov_right))

    m = np.square(mu_left - mu_right).sum()
    s, _ = splin.sqrtm(np.dot(cov_left, cov_right), disp=False)
    if not np.isfinite(s).all():
        logging.rank_zero_warning(
            "Singular product detected during FID calculation. "
            "Adding %s to diagonal of covariance estimations.",
            eps,
        )
        offset = np.eye(cov_left.shape[0]) * eps
        s, _ = splin.sqrtm(
            np.dot((cov_left + offset), (cov_right + offset)),
            disp=False,
        )

    if np.iscomplexobj(s):
        logging.rank_zero_warning(
            "Complex component detected in matrix square root "
            "of the product of covariance matrices during FID calculation."
        )

    out = m + np.trace(cov_left + cov_right - 2 * s)

    return np.real(out)


def _process_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    r"""Preprocess and resize the image for FID.

    .. note::

        This is adapted from the original image preprocessing in `clean-fid`:
        `https://github.com/GaParmar/clean-fid/blob/main/cleanfid/resize.py`

    Args:
        image (npt.NDArray[np.uint8]): The input image to be processed.

    Returns:
        The processed and resized image as a NumPy array.
    """

    def __resize(channel: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        pil_image = Image.fromarray(channel.astype(np.float32), mode="F")
        pil_image = pil_image.resize((299, 299), Image.Resampling.BICUBIC)
        out = np.asarray(pil_image).clip(0, 255)
        out = out.astype(np.uint8).reshape(299, 299, 1)
        return out

    out = np.concatenate(
        [__resize(np.array(image[..., c])) for c in range(image.shape[-1])],
        axis=-1,
    )

    return out


@functools.partial(jax.jit, static_argnames="target_shape")
def _tf_legacy_resize_bilinear(
    image: jax.Array,
    shape: typing.Sequence[int],
) -> jax.Array:
    r"""Reproduces TensorFlow 1.x's ``ResizeBilinear`` with:

    - `half_pixel_centers=False`, and
    - `align_corners=False`

    This is critical for running FID score with parameters converted directly
    from the ``TensorFlow`` checkpoint at `http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz`.

    Args:
        image (jax.Array): Input array of shape ``(height, width, channels)``.
        shape (Sequence[int]): Target shape ``(target_height, target_width)``.

    Returns:
        Resized image of shape ``(target_height, target_width, channels)``.
    """
    in_h, in_w = image.shape[-3:-1]
    out_h, out_w = shape[0:2]

    # scale calculation (align_corners=False)
    scale_y = in_h / out_h
    scale_x = in_w / out_w

    # X coordinates
    grid_x = jnp.arange(out_w, dtype=jnp.float32)
    grid_x = grid_x * scale_x
    grid_x_lo = jnp.floor(grid_x).astype(jnp.int32)
    grid_x_hi = jnp.clip(grid_x_lo + 1, 0, in_w - 1)
    grid_x_lo = jnp.clip(grid_x_lo, 0, in_w - 1)
    grid_dx = grid_x - grid_x_lo.astype(jnp.float32)

    # Y coordinates
    grid_y = jnp.arange(out_h, dtype=jnp.float32)
    grid_y = grid_y * scale_y
    grid_y_lo = jnp.floor(grid_y).astype(jnp.int32)
    grid_y_hi = jnp.clip(grid_y_lo + 1, 0, in_h - 1)
    grid_y_lo = jnp.clip(grid_y_lo, 0, in_h - 1)
    grid_dy = grid_y - grid_y_lo.astype(jnp.float32)

    # Gather the four corners
    in_00 = image[grid_y_lo, :, :][:, grid_x_lo, :]
    in_01 = image[grid_y_lo, :, :][:, grid_x_hi, :]
    in_10 = image[grid_y_hi, :, :][:, grid_x_lo, :]
    in_11 = image[grid_y_hi, :, :][:, grid_x_hi, :]

    # Bilinear interpolation
    in_0 = in_00 + (in_01 - in_00) * grid_dx.reshape(1, out_w, 1)
    in_1 = in_10 + (in_11 - in_10) * grid_dx.reshape(1, out_w, 1)
    out = in_0 + (in_1 - in_0) * grid_dy.reshape(out_h, 1, 1)

    return out


class FrechetInceptionDistance:
    r"""Computes the Fréchet Inception Distance (FID) score.

    Args:
        dataset (datasets.Dataset): The reference dataset used to compute
            the reference statistics.
        image_key (str, optional): The column name in the dataset that
            contains the images. Default is `"image"`.
        batch_size (int, optional): The batch size for processing images.
            Default is `32`.
        mode (str, optional): The mode of image processing to use. Either
            `"tensorflow"` or `"clean"`. Default is `"tensorflow"`.
    """

    _mode: str
    _ref_mu: npt.NDArray[np.float64]
    _ref_cov: npt.NDArray[np.float64]

    def __init__(
        self,
        dataset: datasets.Dataset,
        image_key: str = "image",
        batch_size: int = 32,
        mode: str = "jax",
    ) -> None:
        self._batch_size = batch_size

        if mode not in ["clean", "jax", "tensorflow"]:
            raise ValueError(
                f"Unsupported FID mode '{mode}'. "
                "Supported modes are 'clean', 'jax', and 'tensorflow'."
            )
        self._mode = mode

        # NOTE: The original FID InceptionV3 variant uses 1,008 output classes
        # Do not change this unless the weights are updated.
        self._model = inception.InceptionV3Network(
            num_classes=1_008,
            last_block_max_pool=True,
            with_aux_logits=False,
            # NOTE: the following lines are crucial for best reproducibility
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )

        # download converted inception v3 weights
        logging.rank_zero_info("Downloading FID Inception V3 weights...")
        with open(
            hf_hub_download(
                repo_id="ChocolateDave/fid-inception-v3",
                filename="fid_inception_v3.msgpack",
                token=os.getenv("HF_TOKEN", None),
                revision="94a8c495414d53da905bab7a40284f02a931d937",
            ),
            mode="rb",
        ) as f:
            self._variables = serialization.msgpack_restore(f.read())
        self._compute_feat = jax.jit(
            functools.partial(self.extract_features, model=self._model),
        )

        # compute reference statistics
        with tqdm_logging.logging_redirect_tqdm():
            if jax.process_index() == 0:
                pbar = tqdm.tqdm(
                    total=len(dataset),
                    desc="Processing reference images...",
                    unit="images",
                )
            else:
                pbar = None

            ref_images = []
            for item in dataset:
                assert isinstance(item, typing.Dict)
                image = item.get(image_key, None)
                if image is None:
                    raise ValueError(f"'{image_key}' not found in dataset.")
                image = self.process(np.array(image))
                ref_images.append(image)
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

            ref_features = []
            if jax.process_index() == 0:
                pbar = tqdm.tqdm(
                    total=len(range(0, len(ref_images), self._batch_size)),
                    desc="Extracting reference features...",
                    unit="batches",
                )
            else:
                pbar = None

            for i in range(0, len(ref_images), self._batch_size):
                batch_images = jnp.array(ref_images[i : i + self._batch_size])
                feats = self._compute_feat(
                    batch_images,
                    params=self._variables["params"],
                    batch_stats=self._variables["batch_stats"],
                )
                ref_features.append(feats)
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

        ref_feats = np.concatenate(ref_features, axis=0).astype(np.float64)
        self._ref_mu = np.mean(ref_feats, axis=0)
        self._ref_cov = np.cov(ref_feats, rowvar=False)

    def __call__(self, images: npt.NDArray[np.uint8]) -> npt.NDArray:
        r"""Computes the FID score between the given images and the reference.

        Args:
            images (npt.NDArray[np.uint8]): A sequence of images to compute the
                FID score against the reference training dataset statistics.
                The images should be of `uint8` type ranged between `[0, 255]`.

        Returns:
            The FID score as a scalar array.
        """
        # sanity checks
        chex.assert_type(images, jnp.uint8)
        chex.assert_rank(images, 4)

        if jax.process_index() == 0:
            pbar = tqdm.tqdm(
                total=len(images),
                desc="Processing sampled images...",
                unit="images",
            )
        else:
            pbar = None
        processed_images = []
        with tqdm_logging.logging_redirect_tqdm():
            for image in images:
                image = self.process(image)
                processed_images.append(image)
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

        if jax.process_index() == 0:
            pbar = tqdm.tqdm(
                total=len(range(0, len(processed_images), self._batch_size)),
                desc="Extracting sampled features...",
                unit="batches",
            )
        else:
            pbar = None
        samp_features = []
        for i in range(0, len(processed_images), self._batch_size):
            batch_images = jnp.array(
                processed_images[i : i + self._batch_size]
            )
            feats = self._compute_feat(
                batch_images,
                params=self._variables["params"],
                batch_stats=self._variables["batch_stats"],
            )
            samp_features.append(feats)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        samp_feats = np.concatenate(samp_features, axis=0).astype(np.float64)
        samp_mu = np.mean(samp_feats, axis=0)
        samp_cov = np.cov(samp_feats, rowvar=False)
        fid_score = _frechet_distance(
            mu_left=samp_mu,
            cov_left=samp_cov,
            mu_right=self._ref_mu,
            cov_right=self._ref_cov,
        )

        return fid_score

    def process(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        r"""Processes and resizes images for FID computation.

        Args:
            images (npt.NDArray[np.uint8]): The image of shape ``(H, W, C)``.
                The images should be of `uint8` type ranged between `[0, 255]`.

        Returns:
            The processed and resized images as a NumPy array.
        """
        if self._mode == "clean":
            return _process_image(image)
        elif self._mode == "jax":
            return np.array(
                jax.image.resize(
                    image=jnp.array(image, dtype=jnp.uint8),
                    shape=(299, 299, 3),
                    method=jax.image.ResizeMethod.LINEAR,
                    antialias=False,
                )
            )
        elif self._mode == "tensorflow":
            return np.array(
                _tf_legacy_resize_bilinear(
                    image=jnp.asarray(image).astype(jnp.float32),
                    shape=(299, 299),
                ),
                dtype=np.uint8,
            )
        else:
            raise ValueError(f"Unsupported FID mode '{self._mode}'.")

    @property
    def ref_mu(self) -> npt.NDArray[np.float64]:
        """npt.NDArray: The reference mean vector of shape `(D,)`."""
        return self._ref_mu

    @property
    def ref_cov(self) -> npt.NDArray[np.float64]:
        """npt.NDArray: The reference covariance matrix of shape `(D, D)`."""
        return self._ref_cov

    @staticmethod
    def extract_features(
        inputs: jax.Array,
        model: inception.InceptionV3Network,
        params: jaxtyping.PyTree,
        batch_stats: jaxtyping.PyTree,
    ) -> jax.Array:
        r"""Computes the feature map from the deepest layer of Inception V3."""
        out = 0.0078125 * (jnp.astype(inputs, jnp.float32) - 128.0)
        out, _ = model.apply(
            variables={"params": params, "batch_stats": batch_stats},
            inputs=out,
            deterministic=True,
            with_head=False,
        )
        return out
