REP 40 --- Variance-Aware Mean Flows
====================================

**Author**: Juanwu Lu
**Status**: Active
**Type**: Standard
**Created**: 09-May-2026

.. contents:: Table of Contents

Motivation
----------

`MeanFlow <https://arxiv.org/abs/2505.13447>` promises distillation-free, one-step generation by enforcing *a total-derivative identity* on the average velocity field. In practice, however, the training is unstable: the loss is *non-decreasing*, *high-variance*, and prone to collapse. Concurrent work diagnoses this from different perspectives, including *gradient conflict*, *Jacobian amplification*, and *curvature bottleneck*. But they generally treat them as separate problems.

FID Reproduction for Fair Comparison
------------------------------------

One `identified issues <https://github.com/Gsunshine/meanflow/issues/6>` with the original MeanFlow repository is its FID calculation. Therefore, for proper debugging during developments and fair comparison to the existing methods, we reproduce the FID metric based on existing implementation.

.. code-block:: python
   :linenos:
   :caption: Use FID implementation.

   from src.projects.generative.tools import fid

   class _LazyFIDMetric:
    r"""Defers FID metric initialization to first ``__call__``."""

    def __init__(self, metric_config: fdl.Config) -> None:
        self._config = metric_config
        self._metric: typing.Optional[fid.FrechetInceptionDistance] = None

    def __call__(self, **kwargs):
        if self._metric is None:
            logging.rank_zero_info("Lazily initializing FID metric...")
            self._metric = fdl.build(self._config)
            if not isinstance(self._metric, fid.FrechetInceptionDistance):
                raise TypeError(
                    "Expected FrechetInceptionDistance, "
                    f"got {type(self._metric)}"
                )
            logging.rank_zero_info("FID metric initialized.")
        return self._metric(**kwargs)

    ...

    if jax.process_index() == 0 and fid_metric is not None:
        # NOTE: only compute FID metric on process 0
        fid_score = fid_metric(images=images[0:50_000])
        outputs.scalars = {"fid": fid_score}

We validate our reproduction by training a Denoising Diffusion Probabilistic Models and compared its evaluation results against `the original one reported by Ho et al. <https://arxiv.org/abs/2006.11239>`. When we compute FID with respect to the test set, `the score is 5.23192 <https://wandb.ai/pdt-purdue-university/ddpm/groups/unet_cifar10_20260126_160248/runs/jmfv302j>` compared to the 5.24 reported by the paper. Therefore, we consider our reproduction correct.

Theoretical Frameworks
----------------------

Report of this project is located at `src/proejcts/generative/vamf/report` and on ArXiv. Our theory identified the two distinct role of conditional velocity field in the original MeanFlow loss and how we can balance bias-variance trade-off by finetuning a mixing coefficient between the conditional velocity field and a deterministic field for the tangent in the Jacobian-vector product.
