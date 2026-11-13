"""
Construction utilities for the VQ-VAE training objective.
"""

from __future__ import annotations

from classes.vqvae_objective import (
    VQVAEObjective,
    VQVAEObjectiveSettings,
)
from settings import VQVAEExperimentSettings


def create_objective(
    settings: VQVAEExperimentSettings,
) -> VQVAEObjective:
    """
    Construct the VQ-VAE training objective.

    Parameters
    ----------
    settings:
        Experiment configuration.

    Returns
    -------
    VQVAEObjective
        Configured training objective.

    Notes
    -----
    Gradient-based vector quantization uses the explicit VQ loss.
    EMA-based vector quantization does not use that loss because
    codebook updates are performed by the quantizer's EMA mechanism.
    """
    return VQVAEObjective(
        VQVAEObjectiveSettings(
            beta=settings.commitment_beta,
            use_mse=True,
            use_bce=False,
            use_vq_loss=(
                not settings.use_ema
            ),
            use_variance_loss=(
                not settings.use_ema
                and settings.use_variance
            ),
            gamma_variance=1.0,
        )
    )