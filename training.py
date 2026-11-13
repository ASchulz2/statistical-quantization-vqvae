"""
training.py
===========

Construction utilities for VQ-VAE training.

The public entry point is :func:`create_trainer`.
"""

from __future__ import annotations

from classes.history import (
    History,
    HistorySettings,
)
from classes.vqvae import VQVAE
from classes.vqvae_objective import VQVAEObjective
from classes.vqvae_trainer import (
    VQVAETrainer,
    VQVAETrainerSettings,
)
from settings import VQVAEExperimentSettings
from classes.visualizer import (
    Visualizer,
    VisualizationSettings
)


def create_trainer(
    model: VQVAE,
    objective: VQVAEObjective,
    settings: VQVAEExperimentSettings,
    history: History | None = None,
    checkpoint_name: str | None = None,
) -> VQVAETrainer:
    """
    Construct a VQ-VAE trainer from experiment settings.

    Parameters
    ----------
    model:
        VQ-VAE model to train.

    objective:
        Objective used to compute training and validation losses
        and metrics.

    settings:
        Complete experiment configuration.

    history:
        Optional existing training history. A new history is created
        when ``None`` is supplied.

    checkpoint_name:
        Optional checkpoint filename used for the selected
        best-model checkpoint. When ``None``, the
        ``best_checkpoint_name`` from ``settings`` is used.

    Returns
    -------
    VQVAETrainer
        Configured VQ-VAE trainer.
    """
    if checkpoint_name is None:
        checkpoint_name = (
            settings.best_checkpoint_name
        )

    checkpoint_path = (
        settings.checkpoint_directory
        /
        checkpoint_name
    )

    trainer_settings = VQVAETrainerSettings(
        epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        device=settings.device,
        print_every=settings.print_every,
        save_best_checkpoint=(
            settings.save_best_checkpoint
        ),
        checkpoint_path=checkpoint_path,
        restore_metric=settings.restore_metric,
        restore_best_model=(
            settings.restore_best_model
        ),
        plot_validation_latent_space=(
            settings.plot_validation_latent_space
        )
    )

    if history is None:
        history = History(
            HistorySettings()
        )

    visualizer = Visualizer(
        VisualizationSettings()
    )

    return VQVAETrainer(
        model=model,
        objective=objective,
        settings=trainer_settings,
        history=history,
        visualizer=visualizer
    )