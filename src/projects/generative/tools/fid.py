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

    diff = mu_left - mu_right
    covmean, _ = splin.sqrtm(np.dot(cov_left, cov_right), disp=False)
    if not np.isfinite(covmean).all():
        logging.rank_zero_warning(
            "Singular product detected during FID calculation. "
            "Adding %s to diagonal of covariance estimations.",
            eps,
        )
        offset = np.eye(cov_left.shape[0]) * eps
        covmean, _ = splin.sqrtm(
            np.dot((cov_left + offset), (cov_right + offset)),
            disp=False,
        )

    if np.iscomplexobj(covmean):
        logging.rank_zero_warning(
            "Complex component detected in covmean during FID calculation. "
            "Taking only the real part.",
        )
        covmean = covmean.real

    trm = np.trace(covmean)
    out = diff.dot(diff) + np.trace(cov_left) + np.trace(cov_right) - 2 * trm

    return out


def _process_image(image: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    def __resize(channel: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        pil_image = Image.fromarray(channel, mode="F")
        pil_image = pil_image.resize((299, 299), Image.Resampling.BICUBIC)
        out = np.asarray(pil_image).clip(0, 255).reshape(299, 299, 1)
        return out

    image = np.array(image).astype(np.float32)
    image = np.concatenate(
        [__resize(image[..., c]) for c in range(image.shape[-1])],
        axis=-1,
    )

    return image


class FrechetInceptionDistance:
    r"""Computes the Fréchet Inception Distance (FID) score.

    Args:
        train_dataset (datasets.Dataset): The training dataset used to compute
            the reference statistics.
        image_key (str, optional): The column name in the dataset that
            contains the images. Default is `"image"`.
        batch_size (int, optional): The batch size for processing images.
            Default is `32`.
    """

    _ref_mu: jxt.ArrayLike
    _ref_cov: jxt.ArrayLike

    def __init__(
        self,
        train_dataset: datasets.Dataset,
        image_key: str = "image",
        batch_size: int = 32,
    ) -> None:
        self._batch_size = batch_size
        self._model = inception.InceptionV3(
            num_classes=1_008,
            last_block_max_pool=True,
            with_aux_logits=False,
        )
        logging.rank_zero_info("Downloading FID Inception V3 weights...")
        with open(
            hf_hub_download(
                repo_id="ChocolateDave/fid-inception-v3",
                filename="fid_inception_v3.msgpack",
                token=os.getenv("HF_TOKEN", None),
                revision="bef27900b6b2c46b866b628a86a1c1cedd95a041",
            ),
            mode="rb",
        ) as f:
            self._variables = serialization.msgpack_restore(f.read())
        self._compute_feat = jax.jit(
            functools.partial(self.extract_features, model=self._model),
        )

        with tqdm_logging.logging_redirect_tqdm():
            if jax.process_index() == 0:
                pbar = tqdm.tqdm(
                    total=len(train_dataset),
                    desc="Computing reference statistics...",
                    unit="images",
                )
            else:
                pbar = None

            ref_images = []
            for item in train_dataset:
                assert isinstance(item, typing.Dict)
                image = item.get(image_key, None)
                if image is None:
                    raise ValueError(
                        f"FATAL: Image key '{image_key}' not found in dataset."
                    )
                image = _process_image(image)
                ref_images.append(image)
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

            ref_features = []
            if jax.process_index() == 0:
                pbar = tqdm.tqdm(
                    total=len(range(0, len(ref_images), batch_size)),
                    desc="Extracting training features...",
                    unit="batches",
                )
            else:
                pbar = None

            for i in range(0, len(ref_images), batch_size):
                batch_images = jnp.array(ref_images[i : i + batch_size])
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

        self._ref_mu = jnp.mean(
            jnp.concatenate(ref_features, axis=0),
            axis=0,
        )
        self._ref_cov = jnp.cov(
            jnp.matrix_transpose(jnp.concatenate(ref_features, axis=0)),
        )

    def __call__(self, images: npt.NDArray) -> npt.NDArray:
        r"""Computes the FID score between the given images and the reference.

        Args:
            images (npt.NDArray): A sequence of images to compute the FID score
                against the reference training dataset statistics. The images
                should be of `uint8` type ranged between `[0, 255]`.

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
                image = _process_image(image)
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
        sampled_features = []
        for i in range(0, len(processed_images), self._batch_size):
            batch_images = jnp.array(
                processed_images[i : i + self._batch_size]
            )
            feats = self._compute_feat(
                batch_images,
                params=self._variables["params"],
                batch_stats=self._variables["batch_stats"],
            )
            sampled_features.append(feats)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        samp_mu = jnp.mean(
            jnp.concatenate(sampled_features, axis=0),
            axis=0,
        )
        samp_cov = jnp.cov(
            jnp.matrix_transpose(jnp.concatenate(sampled_features, axis=0)),
        )
        fid_score = _frechet_distance(
            mu_left=samp_mu,
            cov_left=samp_cov,
            mu_right=self._ref_mu,
            cov_right=self._ref_cov,
        )

        return fid_score

    @property
    def ref_mu(self) -> jxt.ArrayLike:
        """ArrayLike: The reference mean vector of shape `(D,)`."""
        return self._ref_mu

    @property
    def ref_cov(self) -> jxt.ArrayLike:
        """ArrayLike: The reference covariance matrix of shape `(D, D)`."""
        return self._ref_cov

    @staticmethod
    def extract_features(
        inputs: jax.Array,
        model: inception.InceptionV3,
        params: jaxtyping.PyTree,
        batch_stats: jaxtyping.PyTree,
    ) -> jax.Array:
        r"""Computes the feature map from the deepest layer of Inception V3."""
        inputs = (jnp.astype(inputs, jnp.float32) - 128.0) / 128.0
        feat, _ = model.apply(
            variables={"params": params, "batch_stats": batch_stats},
            inputs=inputs,
            deterministic=True,
            with_head=False,
        )
        return feat
