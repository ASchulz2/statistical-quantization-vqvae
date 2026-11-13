"""
model.py
========

Construction utilities for the VQ-VAE model.

The module keeps experiment-level configuration separate from the
lower-level encoder, decoder, and vector-quantizer settings.

The public entry point is :func:`create_model`.
"""

from __future__ import annotations

from classes.decoder import DecoderSettings
from classes.encoder import EncoderSettings
from classes.vector_quantizer_torch import (
    VectorQuantizerTorchSettings,
)
from settings import VQVAEExperimentSettings

from classes.vqvae import (
    VQVAE,
    VQVAESettings,
)


def create_model(
    input_dim: int,
    settings: VQVAEExperimentSettings,
) -> VQVAE:
    """
    Construct a configured VQ-VAE.

    Parameters
    ----------
    input_dim:
        Dimensionality of the flattened input representation.

    settings:
        Experiment-level VQ-VAE configuration.

    Returns
    -------
    VQVAE
        Configured VQ-VAE model.

    Raises
    ------
    ValueError
        If ``input_dim`` is not positive.

    Notes
    -----
    The experiment-level ``commitment_beta`` parameter is mapped to
    the lower-level vector quantizer's ``commitment_cost`` parameter.

    The encoder and decoder architectures are taken directly from
    ``settings.encoder_hidden_dims`` and ``settings.decoder_hidden_dims``.
    """
    if input_dim <= 0:
        raise ValueError(
            "input_dim must be positive."
        )

    encoder_settings = EncoderSettings(
        input_dim=input_dim,
        hidden_dims=settings.encoder_hidden_dims,
        latent_dim=settings.latent_dim,
        device=settings.device,
    )

    decoder_settings = DecoderSettings(
        latent_dim=settings.latent_dim,
        hidden_dims=settings.decoder_hidden_dims,
        output_dim=input_dim,
        device=settings.device,
    )

    quantizer_settings = VectorQuantizerTorchSettings(
        num_embeddings=settings.num_embeddings,
        embedding_dim=settings.latent_dim,
        commitment_cost=settings.commitment_beta,
        device=settings.device,
        use_ema=settings.use_ema,
        use_variance=settings.use_variance,
        ema_decay=settings.ema_decay,
    )

    model_settings = VQVAESettings(
        encoder=encoder_settings,
        decoder=decoder_settings,
        quantizer=quantizer_settings,
    )

    return VQVAE(
        model_settings
    )