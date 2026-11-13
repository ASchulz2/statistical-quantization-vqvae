"""
Checkpoint persistence utilities for VQ-VAE experiments.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from classes.vqvae import VQVAE
from settings import VQVAEExperimentSettings


def _serialize_settings(
    settings: VQVAEExperimentSettings,
) -> dict[str, Any]:
    """
    Convert experiment settings into a checkpoint-safe dictionary.
    """
    values = asdict(
        settings
    )

    for key, value in values.items():

        if isinstance(
            value,
            Path,
        ):
            values[key] = str(value)

    return values


def save_final_checkpoint(
    model: VQVAE,
    history,
    trainer,
    evaluation_metrics: dict,
    evaluation_split: str,
    dataset_name: str,
    settings: VQVAEExperimentSettings,
) -> Path:
    """
    Save the final selected VQ-VAE checkpoint.

    Parameters
    ----------
    model:
        Selected trained VQ-VAE.

    history:
        Training and validation history.

    trainer:
        Trainer containing model-selection statistics.

    evaluation_metrics:
        Metrics obtained from the evaluated dataset split.

    evaluation_split:
        Name of the evaluated dataset split, for example
        ``"validation"`` or ``"test"``.

    dataset_name:
        Name of the dataset used for the experiment.

    settings:
        Experiment configuration.

    Returns
    -------
    Path
        Path to the saved final checkpoint.
    """
    output_directory = (
        settings.checkpoint_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "history": (
            history.as_dict()
        ),
        "settings": {
            **_serialize_settings(settings),
            "dataset": dataset_name,
        },
        "model_selection": {
            "criterion": (
                settings.restore_metric
            ),
            "best_val_distortion": (
                trainer.best_val_distortion
            ),
            "best_val_distortion_epoch": (
                trainer.best_val_distortion_epoch
            ),
            "best_val_loss": (
                trainer.best_val_loss
            ),
            "best_val_loss_epoch": (
                trainer.best_val_loss_epoch
            ),
        },
        "evaluation": {
            "split": evaluation_split,
            "metrics": {
                "distortion": (
                    evaluation_metrics["distortion"]
                ),
                "distortion_db": (
                    evaluation_metrics["distortion_db"]
                ),
                "rmse": (
                    evaluation_metrics["rmse"]
                ),
                "psnr": (
                    evaluation_metrics["psnr"]
                ),
                "rate": (
                    evaluation_metrics["rate"]
                ),
                "entropy": (
                    evaluation_metrics["entropy"]
                ),
                "perplexity": (
                    evaluation_metrics["perplexity"]
                ),
                "utilization": (
                    evaluation_metrics["utilization"]
                ),
                "active_codewords": (
                    evaluation_metrics["active_codewords"]
                ),
                "dead_codewords": (
                    evaluation_metrics["dead_codewords"]
                ),
            },
        },
    }

    output_path = (
        output_directory
        /
        f"vqvae_{dataset_name}.pt"
    )

    torch.save(
        checkpoint,
        output_path,
    )

    return output_path

