"""
Tests for experiment result creation and persistence.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from classes.experiment_results import ExperimentResult
from experiment_results import (
    create_experiment_result,
    save_experiment_result,
)
from settings import VQVAEExperimentSettings


def create_test_trainer() -> SimpleNamespace:
    """Create minimal trainer state for result tests."""
    return SimpleNamespace(
        best_val_loss=1.25,
        best_val_distortion=0.75,
        best_val_loss_epoch=3,
        best_val_distortion_epoch=4,
    )


def create_test_metrics() -> dict:
    """Create representative test metrics."""
    return {
        "distortion": 0.10,
        "distortion_db": -10.0,
        "rmse": 0.20,
        "psnr": 14.0,
        "rate": 3.25,
        "perplexity": 12.5,
        "utilization": 0.75,
        "entropy": 3.64,
        "active_codewords": 768,
        "dead_codewords": 256,
    }


def test_create_experiment_result() -> None:
    settings = VQVAEExperimentSettings(
        seed=89,
        epochs=5,
        latent_dim=2,
        num_embeddings=1024,
        train_batch_size=128,
        learning_rate=1e-3,
        commitment_beta=1.0,
        use_ema=True,
        use_variance=False,
    )

    result = create_experiment_result(
        dataset_name="mnist",
        runtime=12.5,
        trainer=create_test_trainer(),
        evaluation_metrics=create_test_metrics(),
        settings=settings,
    )

    assert isinstance(
        result,
        ExperimentResult,
    )

    assert result.dataset == "mnist"
    assert result.method == "Statistical VQ-VAE"
    assert result.predictive is False

    assert result.latent_dim == 2
    assert result.num_codewords == 1024
    assert result.epochs == 5
    assert result.batch_size == 128
    assert result.learning_rate == 1e-3
    assert result.commitment_beta == 1.0

    assert result.use_ema is True
    assert result.use_variance is False
    assert result.seed == 89
    assert result.runtime == 12.5

    assert result.distortion == 0.10
    assert result.distortion_db == -10.0
    assert result.rmse == 0.20
    assert result.psnr == 14.0

    assert result.rate == 3.25
    assert result.perplexity == 12.5
    assert result.utilization == 0.75
    assert result.entropy == 3.64

    assert result.active_codewords == 768
    assert result.dead_codewords == 256

    assert result.best_val_loss == 1.25
    assert result.best_val_distortion == 0.75
    assert result.best_val_loss_epoch == 3
    assert result.best_val_distortion_epoch == 4


def test_create_experiment_result_requires_best_loss_epoch() -> None:
    trainer = create_test_trainer()
    trainer.best_val_loss_epoch = None

    with pytest.raises(RuntimeError):
        create_experiment_result(
            dataset_name="mnist",
            runtime=1.0,
            trainer=trainer,
            evaluation_metrics=create_test_metrics(),
            settings=VQVAEExperimentSettings(),
        )


def test_create_experiment_result_requires_best_distortion_epoch() -> None:
    trainer = create_test_trainer()
    trainer.best_val_distortion_epoch = None

    with pytest.raises(RuntimeError):
        create_experiment_result(
            dataset_name="mnist",
            runtime=1.0,
            trainer=trainer,
            evaluation_metrics=create_test_metrics(),
            settings=VQVAEExperimentSettings(),
        )


def test_save_experiment_result(
    tmp_path: Path,
) -> None:
    settings = VQVAEExperimentSettings(
        results_path=(
            tmp_path / "results" / "experiments.csv"
        ),
    )

    result = create_experiment_result(
        dataset_name="mnist",
        runtime=1.0,
        trainer=create_test_trainer(),
        evaluation_metrics=create_test_metrics(),
        settings=settings,
    )

    output_path = save_experiment_result(
        result=result,
        settings=settings,
    )

    assert output_path.exists()
    assert output_path == settings.results_path

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 1
    assert rows[0]["dataset"] == "mnist"
    assert (
        rows[0]["method"]
        == "Statistical VQ-VAE"
    )