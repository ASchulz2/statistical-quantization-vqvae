"""
Sampling utilities for statistical VQ-VAE latents.
"""

from __future__ import annotations

import torch

from classes.vqvae import VQVAE


@torch.no_grad()
def sample_quantized_latents(
    model: VQVAE,
    z_e: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """
    Sample latent vectors from the conditional codeword distribution.

    Parameters
    ----------
    model:
        VQ-VAE containing the trained statistical codebook.

    z_e:
        Continuous encoder representations used to determine
        codeword assignments.

    noise_scale:
        Scale applied to the estimated conditional standard deviation.
        A value of ``0.0`` produces deterministic codeword means.

    Returns
    -------
    torch.Tensor
        Sampled quantized latent representations.

    Raises
    ------
    ValueError
        If ``noise_scale`` is negative.

    AttributeError
        If variance sampling is requested but the quantizer does
        not expose ``embedding_logvar``.
    """
    if noise_scale < 0.0:
        raise ValueError(
            "noise_scale must be non-negative."
        )

    quantizer = model.quantizer

    indices = quantizer.assign(
        z_e
    )

    means = quantizer.embedding_mean(
        indices
    )

    #
    # Deterministic path.
    #

    if (
        noise_scale == 0.0
        or not quantizer.settings.use_variance
    ):
        return means

    #
    # Variance must be available.
    #

    if not hasattr(
        quantizer,
        "embedding_logvar",
    ):
        raise AttributeError(
            "Variance sampling requested, but the "
            "quantizer does not expose "
            "'embedding_logvar'."
        )

    logvar = quantizer.embedding_logvar(
        indices
    )

    variance = torch.exp(
        logvar
    )

    variance = torch.clamp(
        variance,
        min=0.0,
    )

    std = torch.sqrt(
        variance + 1e-8
    )

    epsilon = torch.randn_like(
        std
    )

    return (
        means
        +
        noise_scale
        *
        std
        *
        epsilon
    )


@torch.no_grad()
def sample_codeword_latents(
    model: VQVAE,
    indices: torch.Tensor,
    noise_scale: float,
    n_samples: int,
) -> torch.Tensor:
    """
    Sample latent vectors from fixed codeword assignments.

    Returns
    -------
    torch.Tensor
        Tensor with shape
        ``(n_samples, n_observations, latent_dim)``.
    """
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative.")

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    means = model.quantizer.embedding_mean(indices)

    if noise_scale == 0.0:
        return (
            means.unsqueeze(0)
            .expand(n_samples, -1, -1)
            .clone()
        )

    if not model.quantizer.settings.use_variance:
        return (
            means.unsqueeze(0)
            .expand(n_samples, -1, -1)
            .clone()
        )

    if not hasattr(model.quantizer, "embedding_logvar"):
        raise AttributeError(
            "Statistical sampling requires 'embedding_logvar'."
        )

    logvar = model.quantizer.embedding_logvar(indices)
    variance = torch.clamp(torch.exp(logvar), min=0.0)
    std = torch.sqrt(variance + 1e-8)

    epsilon = torch.randn(
        (
            n_samples,
            indices.shape[0],
            means.shape[-1],
        ),
        device=means.device,
        dtype=means.dtype,
    )

    return (
        means.unsqueeze(0)
        + noise_scale * std.unsqueeze(0) * epsilon
    )


@torch.no_grad()
def sample_from_codebook(
    model: VQVAE,
    probabilities: torch.Tensor,
    n_samples: int,
    noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample codeword indices from an empirical prior and then sample
    continuous latent vectors from the corresponding statistical codewords.

    The generative process is

        k ~ p(k)

        z = mu_k + alpha * sigma_k * epsilon

        epsilon ~ N(0, I)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Generated latent vectors and their sampled codeword indices.
    """
    if noise_scale < 0.0:
        raise ValueError(
            "noise_scale must be non-negative."
        )

    if n_samples < 1:
        raise ValueError(
            "n_samples must be positive."
        )

    if probabilities.ndim != 1:
        raise ValueError(
            "probabilities must be one-dimensional."
        )

    if probabilities.numel() != (
        model.quantizer.settings.num_embeddings
    ):
        raise ValueError(
            "Probability vector length must match "
            "the number of codewords."
        )

    probabilities = probabilities.to(
        device=model.quantizer.embedding_mean.weight.device,
        dtype=torch.float32,
    )

    if torch.any(probabilities < 0.0):
        raise ValueError(
            "probabilities must be non-negative."
        )

    total_probability = probabilities.sum()

    if not torch.isfinite(total_probability):
        raise ValueError(
            "probabilities must contain finite values."
        )

    if total_probability <= 0.0:
        raise ValueError(
            "probabilities must have a positive sum."
        )

    probabilities = (
        probabilities / total_probability
    )

    indices = torch.multinomial(
        probabilities,
        n_samples,
        replacement=True,
    )

    means = model.quantizer.embedding_mean(
        indices
    )

    if noise_scale == 0.0:
        return means, indices

    if not model.quantizer.settings.use_variance:
        raise RuntimeError(
            "Stochastic generation requires a model "
            "trained with variance estimation enabled."
        )

    if not hasattr(
        model.quantizer,
        "embedding_logvar",
    ):
        raise AttributeError(
            "Statistical sampling requires 'embedding_logvar'."
        )

    logvar = model.quantizer.embedding_logvar(
        indices
    )

    variance = torch.clamp(
        torch.exp(logvar),
        min=0.0,
    )

    std = torch.sqrt(
        variance + 1e-8
    )

    epsilon = torch.randn_like(
        std
    )

    latents = (
        means
        + noise_scale * std * epsilon
    )

    return latents, indices