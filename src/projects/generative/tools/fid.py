import functools
import io
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
import tensorflow as tf
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


class FrechetInceptionDistance:
    r"""Computes the Fréchet Inception Distance (FID) score.

    Args:
        dataset (datasets.Dataset): The reference dataset used to compute
            the reference statistics. Ignored when ``ref_cache_path``
            points to an existing ``.npz`` file.
        image_key (str, optional): The column name in the dataset that
            contains the images. Default is `"image"`.
        batch_size (int, optional): The batch size for processing images.
            Default is `32`.
        mode (str, optional): The mode of image processing to use. Either
            `"tensorflow"` or `"clean"`. Default is `"tensorflow"`.
        ref_cache_path (str, optional): Path to a ``.npz`` file for
            caching reference statistics (mu, cov).  Supports local
            paths and ``gs://`` URIs.  If the file exists, statistics
            are loaded directly and the dataset is never opened.  If
            the file does not exist, statistics are computed from
            ``dataset`` and then saved to this path.
    """

    _mode: str
    _ref_mu: npt.NDArray[np.float64]
    _ref_cov: npt.NDArray[np.float64]

    def __init__(
        self,
        dataset: typing.Union[
            datasets.Dataset,
            datasets.IterableDataset,
            typing.Callable,
            None,
        ] = None,
        image_key: str = "image",
        batch_size: int = 32,
        mode: str = "tensorflow",
        ref_cache_path: typing.Optional[str] = None,
    ) -> None:
        self._batch_size = batch_size

        if mode not in ["tensorflow", "clean"]:
            raise ValueError(
                f"Unsupported FID mode '{mode}'. "
                "Supported modes are 'tensorflow' and 'clean'."
            )
        self._mode = mode

        # NOTE: The original FID InceptionV3 variant uses 1,008 output classes
        # Do not change this unless the weights are updated.
        self._model = inception.InceptionV3(
            num_classes=1_008,
            last_block_max_pool=True,
            with_aux_logits=False,
        )

        # download converted inception v3 weights
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

        # load or compute reference statistics
        cached = self._try_load_cache(ref_cache_path)
        if cached is not None:
            self._ref_mu, self._ref_cov = cached
        else:
            if callable(dataset) and not isinstance(
                dataset, (datasets.Dataset, datasets.IterableDataset)
            ):
                dataset = dataset()
            if dataset is None:
                raise ValueError(
                    "No cached reference statistics found at "
                    f"'{ref_cache_path}' and no dataset provided."
                )
            self._ref_mu, self._ref_cov = self._compute_ref_stats(
                dataset, image_key
            )
            if ref_cache_path is not None:
                logging.rank_zero_info(
                    "Saving FID reference statistics to %s",
                    ref_cache_path,
                )
                buf = io.BytesIO()
                np.savez(buf, mu=self._ref_mu, cov=self._ref_cov)
                buf.seek(0)
                with tf.io.gfile.GFile(ref_cache_path, "wb") as fh:
                    fh.write(buf.read())

    @staticmethod
    def _try_load_cache(
        path: typing.Optional[str],
    ) -> typing.Optional[
        typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ]:
        """Load cached reference statistics from *path*.

        Returns ``(mu, cov)`` if the cache exists, ``None`` otherwise.
        """
        if path is None:
            return None
        try:
            with tf.io.gfile.GFile(path, "rb") as fh:
                data = np.load(fh)
                mu = data["mu"].astype(np.float64)
                cov = data["cov"].astype(np.float64)
            logging.rank_zero_info(
                "Loaded cached FID reference statistics from %s", path
            )
            return mu, cov
        except tf.errors.NotFoundError:
            return None

    def _compute_ref_stats(
        self,
        dataset: typing.Union[datasets.Dataset, datasets.IterableDataset],
        image_key: str,
    ) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute reference statistics by streaming through the dataset.

        Processes images in batches and accumulates feature
        statistics online, avoiding loading the full dataset
        into memory.

        Returns:
            Tuple of ``(mu, cov)`` arrays.
        """
        feat_dim = 2048
        n = 0
        sum_f = np.zeros(feat_dim, dtype=np.float64)
        sum_ff = np.zeros((feat_dim, feat_dim), dtype=np.float64)
        batch_buf: typing.List[npt.NDArray[np.uint8]] = []

        try:
            total = len(dataset)
        except TypeError:
            total = None

        with tqdm_logging.logging_redirect_tqdm():
            if jax.process_index() == 0:
                pbar = tqdm.tqdm(
                    total=total,
                    desc="Computing reference statistics...",
                    unit="images",
                )
            else:
                pbar = None

            for item in dataset:
                assert isinstance(item, typing.Dict)
                image = item.get(image_key, None)
                if image is None:
                    raise ValueError(f"'{image_key}' not in dataset.")
                batch_buf.append(self.process(np.array(image)))
                if pbar is not None:
                    pbar.update(1)

                if len(batch_buf) >= self._batch_size:
                    feats = np.asarray(
                        self._compute_feat(
                            jnp.array(batch_buf),
                            params=self._variables["params"],
                            batch_stats=self._variables["batch_stats"],
                        )
                    ).astype(np.float64)
                    n += feats.shape[0]
                    sum_f += feats.sum(axis=0)
                    sum_ff += feats.T @ feats
                    batch_buf = []

            # flush remaining images
            if batch_buf:
                feats = np.asarray(
                    self._compute_feat(
                        jnp.array(batch_buf),
                        params=self._variables["params"],
                        batch_stats=self._variables["batch_stats"],
                    )
                ).astype(np.float64)
                n += feats.shape[0]
                sum_f += feats.sum(axis=0)
                sum_ff += feats.T @ feats

            if pbar is not None:
                pbar.close()

        mu = sum_f / n
        cov = (sum_ff - n * np.outer(mu, mu)) / (n - 1)
        return mu, cov

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

    def process(self, images: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        r"""Processes and resizes images for FID computation.

        Args:
            images (npt.NDArray[np.uint8]): A sequence of images to be processed.
                The images should be of `uint8` type ranged between `[0, 255]`.

        Returns:
            The processed and resized images as a NumPy array.
        """
        if self._mode == "clean":
            return _process_image(images)
        elif self._mode == "tensorflow":
            return np.array(
                jax.image.resize(
                    jnp.array(images, dtype=np.uint8),
                    shape=(299, 299, 3),
                    method=jax.image.ResizeMethod.LINEAR,
                    antialias=False,
                )
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
