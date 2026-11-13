"""
Experiment result creation and persistence.

This module converts a completed VQ-VAE experiment into the
canonical :class:`ExperimentResult` representation and persists
that result as CSV.
"""

from __future__ import annotations

from pathlib import Path

from classes.experiment_results import (
    ExperimentResult,
    ExperimentResults,
    ExperimentResultsSettings,
)
from classes.vqvae_trainer import VQVAETrainer
from settings import VQVAEExperimentSettings


def create_experiment_result(
    dataset_name: str,
    runtime: float,
    trainer: VQVAETrainer,
    evaluation_metrics: dict,
    settings: VQVAEExperimentSettings,
) -> ExperimentResult:
    """
    Create the canonical result record for a completed experiment.

    Parameters
    ----------
    dataset_name:
        Dataset used by the experiment.

    runtime:
        Experiment runtime in seconds.

    trainer:
        Completed VQ-VAE trainer containing validation-selection
        information.

    evaluation_metrics:
        Metrics and statistics obtained from the evaluated dataset
        split. This may contain validation-set or test-set metrics.

    settings:
        Complete experiment configuration.

    Returns
    -------
    ExperimentResult
        Canonical experiment result record.

    Raises
    ------
    RuntimeError
        If best validation epochs are unavailable.
    """
    if trainer.best_val_loss_epoch is None:
        raise RuntimeError(
            "Best validation-loss epoch is unavailable."
        )

    if trainer.best_val_distortion_epoch is None:
        raise RuntimeError(
            "Best validation-distortion epoch is unavailable."
        )

    return ExperimentResult(
        dataset=dataset_name,
        method="Statistical VQ-VAE",
        predictive=False,
        latent_dim=settings.latent_dim,
        num_codewords=settings.num_embeddings,
        epochs=settings.epochs,
        batch_size=settings.train_batch_size,
        learning_rate=settings.learning_rate,
        commitment_beta=settings.commitment_beta,
        use_ema=settings.use_ema,
        use_variance=settings.use_variance,
        seed=settings.seed,
        runtime=runtime,
        distortion=float(
            evaluation_metrics["distortion"]
        ),
        distortion_db=float(
            evaluation_metrics["distortion_db"]
        ),
        rmse=float(
            evaluation_metrics["rmse"]
        ),
        psnr=float(
            evaluation_metrics["psnr"]
        ),
        rate=float(
            evaluation_metrics["rate"]
        ),
        perplexity=float(
            evaluation_metrics["perplexity"]
        ),
        utilization=float(
            evaluation_metrics["utilization"]
        ),
        entropy=float(
            evaluation_metrics["entropy"]
        ),
        active_codewords=int(
            evaluation_metrics["active_codewords"]
        ),
        dead_codewords=int(
            evaluation_metrics["dead_codewords"]
        ),
        best_val_loss=float(
            trainer.best_val_loss
        ),
        best_val_distortion=float(
            trainer.best_val_distortion
        ),
        best_val_loss_epoch=int(
            trainer.best_val_loss_epoch
        ),
        best_val_distortion_epoch=int(
            trainer.best_val_distortion_epoch
        ),
    )


def save_experiment_result(
    result: ExperimentResult,
    settings: VQVAEExperimentSettings,
) -> Path:
    """
    Append one experiment result to the experiment CSV.

    Parameters
    ----------
    result:
        Completed experiment result.

    settings:
        Experiment configuration containing the results path.

    Returns
    -------
    Path
        Path to the resulting CSV file.
    """
    exporter = ExperimentResults(
        ExperimentResultsSettings(
            output_path=str(
                settings.results_path
            ),
            include_header=True,
        )
    )

    exporter.add(
        result
    )

    return exporter.append_csv()
