"""
Visualization orchestration for VQ-VAE experiments.
"""

from __future__ import annotations

import torch

from matplotlib.figure import Figure

from classes.visualizer import Visualizer
from classes.vqvae import VQVAE
from latent_sampling import sample_quantized_latents
from settings import VQVAEExperimentSettings


def visualize_results(
    model: VQVAE,
    history,
    evaluation_metrics: dict,
    visualizer: Visualizer,
    dataset_name: str,
    settings: VQVAEExperimentSettings,
    evaluation_split: str = "evaluation",
) -> list[Figure]:
    """
    Produce diagnostic visualizations for a completed experiment.

    Parameters
    ----------
    model:
        Selected VQ-VAE model.

    history:
        Training and validation history.

    evaluation_metrics:
        Outputs and latent representations from the evaluated
        dataset split. An optional ``targets`` entry may contain
        one target label per evaluated sample.

    visualizer:
        Visualization backend.

    dataset_name:
        Dataset name used in plot titles.

    settings:
        Experiment configuration controlling visualization behavior.

    evaluation_split:
        Name of the evaluated dataset split, for example
        ``"validation"`` or ``"test"``.

    Returns
    -------
    list[Figure]
        Figures created by visualization methods that return figures.
    """
    figures: list[Figure] = []

    originals_tensor = evaluation_metrics[
        "originals"
    ]

    reconstructions_tensor = evaluation_metrics[
        "reconstructions"
    ]

    latents = evaluation_metrics[
        "latents"
    ]

    targets = evaluation_metrics.get(
        "targets"
    )

    n_images = min(
        settings.n_visualization_images,
        len(originals_tensor),
    )

    originals = originals_tensor[
        :n_images
    ]

    reconstructions = reconstructions_tensor[
        :n_images
    ]

    history_data = history.as_dict()

    split_title = (
        evaluation_split.capitalize()
    )

    #
    # Training / validation curves.
    #

    if hasattr(
        visualizer,
        "plot_training_validation_metrics",
    ):
        metric_figures = visualizer.plot_training_validation_metrics(
            history_data
        )

        figures.extend(
            metric_figures
        )

    else:
        metric_figures = visualizer.plot_metrics(
            history_data
        )

        figures.extend(
            metric_figures
        )

    #
    # Reconstructions.
    #

    reconstruction_figure = (
        visualizer.plot_reconstructions(
            originals,
            reconstructions,
            n_images=n_images,
            n_columns=8,
            title=(
                f"{split_title}-set "
                "reconstructions"
            ),
        )
    )

    figures.append(
        reconstruction_figure
    )

    #
    # Reconstruction residuals.
    #

    if hasattr(
        visualizer,
        "plot_reconstruction_residuals",
    ):
        reconstruction_residual_figure = (
            visualizer.plot_reconstruction_residuals(
                original=originals,
                reconstructed=reconstructions,
                n_images=n_images,
                n_columns=8,
                title=(
                    f"{split_title}-set "
                    "reconstruction residuals"
                ),
            )
        )

        figures.append(
            reconstruction_residual_figure
        )

    #
    # Conditional latent generation.
    #

    z_e = latents[
        :n_images
    ].to(
        settings.device
    )

    with torch.no_grad():

        z_generating = (
            sample_quantized_latents(
                model,
                z_e,
                noise_scale=settings.noise_scale,
            )
        )

        generated = model.decode(
            z_generating
        )

    generation_figure = (
        visualizer.plot_generations(
            images=generated,
            n_images=min(
                settings.n_generative_images,
                len(generated),
            ),
            title=(
                f"{dataset_name.upper()} "
                f"{evaluation_split} "
                "conditional latent generation"
            ),
        )
    )

    figures.append(
        generation_figure
    )

    #
    # Decoded codebook.
    #

    if settings.plot_codebook:

        codebook_figure = (
            visualizer.plot_codebook_images(
                decoder=model.decoder,
                codewords=(
                    model.quantizer.embedding_mean.weight
                ),
                device=settings.device,
                n_columns=32,
                title=(
                    f"{dataset_name.upper()} "
                    "decoded codebook"
                ),
            )
        )

        figures.append(
            codebook_figure
        )

    #
    # Two-dimensional latent representation.
    #

    if settings.latent_dim == 2:

        latent_samples = (
            latents
            .detach()
            .cpu()
            .numpy()
        )

        latent_targets = None

        if targets is not None:
            latent_targets = (
                targets
                .detach()
                .cpu()
                .numpy()
            )

        latent_space_figure = (
            visualizer.plot_latent_space(
                latent_samples,
                codewords=(
                    model.quantizer.embedding_mean.weight
                    .detach()
                    .cpu()
                    .numpy()
                ),
                targets=latent_targets,
                title=(
                    f"{dataset_name.upper()} "
                    f"{evaluation_split} "
                    "latent space"
                ),
            )
        )

        figures.append(
            latent_space_figure
        )

    return figures