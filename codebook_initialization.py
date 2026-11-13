"""
Codebook initialization utilities for VQ-VAE experiments.

The codebook is initialized from encoder representations collected
exclusively from the training dataset.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from classes.vqvae import VQVAE
from settings import VQVAEExperimentSettings


def initialize_codebook(
    model: VQVAE,
    train_loader: Iterable[torch.Tensor],
    settings: VQVAEExperimentSettings,
) -> None:
    """
    Initialize the VQ-VAE codebook from training-set representations.

    Parameters
    ----------
    model:
        VQ-VAE whose vector quantizer will be initialized.

    train_loader:
        Training data loader. Only the first
        ``settings.kmeans_init_batches`` batches are used.

    settings:
        Experiment configuration controlling the number of
        initialization batches and the computation device.

    Raises
    ------
    RuntimeError
        If no latent samples are collected.

    Notes
    -----
    The model is switched to evaluation mode during latent-sample
    collection, and gradient computation is disabled.

    The collected encoder representations are passed to
    ``model.quantizer.initialize_from_samples()``. The actual
    K-Means implementation therefore remains encapsulated inside
    the vector quantizer.
    """
    model.eval()

    latent_samples: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= settings.kmeans_init_batches:
                break

            batch = batch.to(
                settings.device
            )

            z = model.encode(
                batch
            )

            latent_samples.append(
                z.detach()
            )

    if not latent_samples:
        raise RuntimeError(
            "No latent samples were collected "
            "for codebook initialization."
        )

    samples = torch.cat(
        latent_samples,
        dim=0,
    )

    model.quantizer.initialize_from_samples(
        samples
    )