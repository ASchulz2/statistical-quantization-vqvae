from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from classes.history import History
from classes.visualizer import Visualizer
from settings import VQVAEExperimentSettings
from training import create_trainer


class TinyModel(nn.Module):
    """
    Minimal model suitable for constructing a VQVAETrainer.

    The trainer only needs a parameterized module during construction
    because it creates an Adam optimizer from model.parameters().
    """

    def __init__(self) -> None:
        super().__init__()

        self.parameter = nn.Parameter(
            torch.tensor(1.0)
        )


class DummyObjective:
    """Placeholder objective for trainer-construction tests."""


def make_settings(
    tmp_path: Path,
    **overrides,
) -> VQVAEExperimentSettings:
    """Create compact experiment settings for construction tests."""
    values = {
        "device": "cpu",
        "epochs": 3,
        "learning_rate": 2e-3,
        "print_every": 2,
        "save_best_checkpoint": True,
        "restore_metric": "val_loss",
        "restore_best_model": False,
        "checkpoint_directory": tmp_path,
    }

    values.update(overrides)

    return VQVAEExperimentSettings(
        **values
    )


def test_create_trainer_constructs_expected_dependencies(
    tmp_path: Path,
) -> None:
    model = TinyModel()
    objective = DummyObjective()

    trainer = create_trainer(
        model=model,
        objective=objective,
        settings=make_settings(tmp_path),
    )

    assert trainer.model is model
    assert trainer.objective is objective
    assert isinstance(trainer.history, History)
    assert isinstance(trainer.visualizer, Visualizer)


def test_create_trainer_creates_history_when_none_is_supplied(
    tmp_path: Path,
) -> None:
    trainer = create_trainer(
        model=TinyModel(),
        objective=DummyObjective(),
        settings=make_settings(tmp_path),
        history=None,
    )

    assert isinstance(
        trainer.history,
        History,
    )


def test_create_trainer_preserves_existing_history(
    tmp_path: Path,
) -> None:
    history = History()

    trainer = create_trainer(
        model=TinyModel(),
        objective=DummyObjective(),
        settings=make_settings(tmp_path),
        history=history,
    )

    assert trainer.history is history


def test_create_trainer_uses_default_checkpoint_name(
    tmp_path: Path,
) -> None:
    settings = make_settings(
        tmp_path,
        best_checkpoint_name="custom_best.pt",
    )

    trainer = create_trainer(
        model=TinyModel(),
        objective=DummyObjective(),
        settings=settings,
    )

    assert trainer.settings.checkpoint_path == (
        tmp_path / "custom_best.pt"
    )


def test_create_trainer_uses_explicit_checkpoint_name(
    tmp_path: Path,
) -> None:
    settings = make_settings(
        tmp_path,
        best_checkpoint_name="default_best.pt",
    )

    trainer = create_trainer(
        model=TinyModel(),
        objective=DummyObjective(),
        settings=settings,
        checkpoint_name="experiment_best.pt",
    )

    assert trainer.settings.checkpoint_path == (
        tmp_path / "experiment_best.pt"
    )


def test_create_trainer_propagates_training_settings(
    tmp_path: Path,
) -> None:
    settings = make_settings(
        tmp_path,
        epochs=7,
        learning_rate=5e-4,
        print_every=3,
        save_best_checkpoint=False,
        restore_metric="val_distortion",
        restore_best_model=True,
    )

    trainer = create_trainer(
        model=TinyModel(),
        objective=DummyObjective(),
        settings=settings,
    )

    assert trainer.settings.epochs == 7
    assert trainer.settings.learning_rate == 5e-4
    assert trainer.settings.device == "cpu"
    assert trainer.settings.print_every == 3
    assert trainer.settings.save_best_checkpoint is False
    assert trainer.settings.restore_metric == (
        "val_distortion"
    )
    assert trainer.settings.restore_best_model is True


def test_create_trainer_propagates_validation_latent_space_setting(
    tmp_path: Path,
) -> None:
    settings = make_settings(
        tmp_path,
        plot_validation_latent_space=True,
    )

    trainer = create_trainer(
        model=TinyModel(),
        objective=DummyObjective(),
        settings=settings,
    )

    assert (
        trainer.settings.plot_validation_latent_space
        is True
    )


def test_create_trainer_disables_validation_latent_space_by_default(
    tmp_path: Path,
) -> None:
    settings = make_settings(
        tmp_path,
        plot_validation_latent_space=False,
    )

    trainer = create_trainer(
        model=TinyModel(),
        objective=DummyObjective(),
        settings=settings,
    )

    assert (
        trainer.settings.plot_validation_latent_space
        is False
    )