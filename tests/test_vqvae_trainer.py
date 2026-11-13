"""
Tests for VQVAETrainer.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import numpy as np

import classes.vqvae_trainer as trainer_module
from classes.vqvae_trainer import (
    VQVAETrainer,
    VQVAETrainerSettings,
)
from dataclasses import replace

from unittest.mock import Mock


class LatentPlotModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.dummy = nn.Parameter(
            torch.tensor(1.0)
        )

        self.quantizer = SimpleNamespace(
            settings=SimpleNamespace(
                embedding_dim=2,
                num_embeddings=4,
            ),
            embedding_mean=nn.Embedding(
                4,
                2,
            ),
        )

    def encode(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return batch[:, :2]


class RecordingLatentVisualizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def plot_latent_space(
        self,
        samples,
        codewords=None,
        targets=None,
        title="",
    ) -> None:
        self.calls.append(
            {
                "samples": samples.copy(),
                "codewords": (
                    None
                    if codewords is None
                    else codewords.copy()
                ),
                "targets": targets,
                "title": title,
            }
        )

class FakeHistory:
    """
    Minimal history implementation for trainer tests.
    """

    def __init__(self) -> None:
        self.values: list[dict] = []

    def update(
        self,
        metrics: dict,
    ) -> None:
        self.values.append(metrics)

    def as_dict(self) -> dict:
        return {
            "metrics": self.values,
        }


class DummyVQStatistics:
    """
    Deterministic replacement for VQStatistics.
    """

    def __init__(
        self,
        settings,
    ) -> None:
        self.settings = settings

    def analyze(
        self,
        assignments,
        num_embeddings: int,
    ) -> dict:
        return {
            "perplexity": 1.0,
            "utilization": 1.0 / num_embeddings,
            "entropy": 0.0,
            "dead_codewords": num_embeddings - 1,
        }


class TinyModel(nn.Module):
    """
    Small differentiable model used to test the trainer.
    """

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Linear(
            4,
            4,
        )

        self.quantizer = SimpleNamespace(
            settings=SimpleNamespace(
                num_embeddings=4,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        reconstruction = self.network(
            x
        )

        indices = torch.zeros(
            x.shape[0],
            dtype=torch.long,
            device=x.device,
        )

        return SimpleNamespace(
            reconstruction=reconstruction,
            indices=indices,
        )


class FakeObjective:
    """
    Differentiable objective for trainer tests.
    """

    def __call__(
        self,
        batch: torch.Tensor,
        output,
    ):
        loss = torch.mean(
            (
                output.reconstruction
                -
                batch
            ).pow(2)
        )

        zero = loss * 0.0

        return SimpleNamespace(
            loss=loss,
            reconstruction_loss=loss,
            vq_loss=zero,
            commitment_loss=zero,
            variance_loss=zero,
            distortion=loss,
            rmse=torch.sqrt(
                loss.detach()
            ),
            psnr=zero.detach(),
        )


@pytest.fixture
def trainer(
    monkeypatch,
) -> VQVAETrainer:
    """
    Create a trainer using only lightweight test doubles.
    """
    monkeypatch.setattr(
        trainer_module,
        "VQStatistics",
        DummyVQStatistics,
    )

    model = TinyModel()

    settings = VQVAETrainerSettings(
        epochs=2,
        learning_rate=1e-3,
        device="cpu",
        print_every=100,
        save_best_checkpoint=False,
        checkpoint_path=Path("test_best.pt"),
        restore_metric="val_distortion",
        restore_best_model=True,
    )

    return VQVAETrainer(
        model=model,
        objective=FakeObjective(),
        settings=settings,
        history=FakeHistory(),
    )


def test_train_step_updates_model(
    trainer: VQVAETrainer,
) -> None:
    batch = torch.rand(
        4,
        4,
    )

    before = {
        key: value.detach().clone()
        for key, value
        in trainer.model.state_dict().items()
    }

    metrics, indices = trainer.train_step(
        batch
    )

    after = trainer.model.state_dict()

    assert set(metrics) == {
        "loss",
        "reconstruction_loss",
        "vq_loss",
        "commitment_loss",
        "variance_loss",
        "distortion",
        "rmse",
        "psnr",
    }

    assert indices.shape == (
        4,
    )

    assert all(
        not torch.equal(
            before[key],
            after[key],
        )
        for key in before
    )


def test_train_epoch_returns_training_metrics(
    trainer: VQVAETrainer,
) -> None:
    loader = [
        torch.rand(
            4,
            4,
        ),
        torch.rand(
            4,
            4,
        ),
    ]

    metrics = trainer.train_epoch(
        loader
    )

    assert trainer.model.training is True

    assert set(metrics) == {
        "loss",
        "reconstruction_loss",
        "vq_loss",
        "commitment_loss",
        "variance_loss",
        "distortion",
        "rmse",
        "psnr",
        "perplexity",
        "utilization",
        "entropy",
        "dead_codewords",
    }


def test_train_epoch_rejects_empty_loader(
    trainer: VQVAETrainer,
) -> None:
    with pytest.raises(RuntimeError):
        trainer.train_epoch([])


def test_validate_returns_validation_metrics(
    trainer: VQVAETrainer,
) -> None:
    loader = [
        torch.rand(
            4,
            4,
        ),
    ]

    metrics = trainer.validate(
        loader
    )

    assert trainer.model.training is False

    assert set(metrics) == {
        "loss",
        "reconstruction_loss",
        "vq_loss",
        "commitment_loss",
        "variance_loss",
        "distortion",
        "rmse",
        "psnr",
        "perplexity",
        "utilization",
        "entropy",
        "dead_codewords",
    }


def test_validate_does_not_modify_model(
    trainer: VQVAETrainer,
) -> None:
    loader = [
        torch.rand(
            4,
            4,
        ),
    ]

    before = {
        key: value.detach().clone()
        for key, value
        in trainer.model.state_dict().items()
    }

    trainer.validate(
        loader
    )

    after = trainer.model.state_dict()

    for key in before:
        assert torch.equal(
            before[key],
            after[key],
        )


def test_validate_rejects_empty_loader(
    trainer: VQVAETrainer,
) -> None:
    with pytest.raises(RuntimeError):
        trainer.validate([])


def test_update_best_model_states(
    trainer: VQVAETrainer,
) -> None:
    trainer.model.network.weight.data.fill_(
        1.0
    )

    improved = trainer._update_best_model_states(
        epoch=1,
        validation_metrics={
            "loss": 10.0,
            "distortion": 5.0,
        },
    )

    assert improved is True

    assert trainer.best_val_loss == 10.0
    assert trainer.best_val_loss_epoch == 1

    assert trainer.best_val_distortion == 5.0
    assert trainer.best_val_distortion_epoch == 1

    assert trainer.best_loss_model_state is not None
    assert trainer.best_distortion_model_state is not None


def test_model_selection_tracks_both_metrics(
    trainer: VQVAETrainer,
) -> None:
    trainer.model.network.weight.data.fill_(
        1.0
    )

    trainer._update_best_model_states(
        epoch=1,
        validation_metrics={
            "loss": 10.0,
            "distortion": 10.0,
        },
    )

    first_loss_state = {
        key: value.clone()
        for key, value
        in trainer.best_loss_model_state.items()
    }

    #
    # Distortion improves, loss does not.
    #

    trainer.model.network.weight.data.fill_(
        2.0
    )

    improved = trainer._update_best_model_states(
        epoch=2,
        validation_metrics={
            "loss": 11.0,
            "distortion": 5.0,
        },
    )

    assert improved is True

    assert trainer.best_val_loss == 10.0
    assert trainer.best_val_loss_epoch == 1

    assert trainer.best_val_distortion == 5.0
    assert trainer.best_val_distortion_epoch == 2

    #
    # The loss-selected state must remain the epoch-1 state.
    #

    assert torch.equal(
        trainer.best_loss_model_state["network.weight"],
        first_loss_state["network.weight"],
    )

    #
    # The distortion-selected state must be the epoch-2 state.
    #

    assert torch.equal(
        trainer.best_distortion_model_state["network.weight"],
        trainer.model.network.weight.cpu(),
    )


def test_val_loss_can_be_selection_metric(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "VQStatistics",
        DummyVQStatistics,
    )

    settings = VQVAETrainerSettings(
        restore_metric="val_loss",
        device="cpu",
    )

    trainer = VQVAETrainer(
        model=TinyModel(),
        objective=FakeObjective(),
        settings=settings,
        history=FakeHistory(),
    )

    assert trainer._update_best_model_states(
        epoch=1,
        validation_metrics={
            "loss": 10.0,
            "distortion": 20.0,
        },
    )

    #
    # Loss improves.
    #

    trainer.model.network.weight.data.fill_(
        2.0
    )

    assert trainer._update_best_model_states(
        epoch=2,
        validation_metrics={
            "loss": 5.0,
            "distortion": 25.0,
        },
    )

    assert trainer.best_val_loss == 5.0
    assert trainer.best_val_loss_epoch == 2

    assert trainer.best_val_distortion == 20.0
    assert trainer.best_val_distortion_epoch == 1


def test_restore_best_restores_selected_distortion_model(
    trainer: VQVAETrainer,
) -> None:
    trainer.model.network.weight.data.fill_(
        1.0
    )

    trainer._update_best_model_states(
        epoch=1,
        validation_metrics={
            "loss": 10.0,
            "distortion": 10.0,
        },
    )

    trainer.model.network.weight.data.fill_(
        2.0
    )

    trainer._update_best_model_states(
        epoch=2,
        validation_metrics={
            "loss": 11.0,
            "distortion": 5.0,
        },
    )

    trainer.model.network.weight.data.fill_(
        99.0
    )

    trainer.restore_best()

    assert torch.allclose(
        trainer.model.network.weight,
        torch.full_like(
            trainer.model.network.weight,
            2.0,
        ),
    )


def test_save_checkpoint(
    trainer: VQVAETrainer,
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path / "best.pt"
    )

    trainer.settings = replace(
        trainer.settings,
        save_best_checkpoint=True,
        checkpoint_path=checkpoint_path,
    )

    train_metrics = {
        "loss": 1.0,
    }

    validation_metrics = {
        "loss": 2.0,
        "distortion": 3.0,
    }

    trainer._update_best_model_states(
        epoch=1,
        validation_metrics=validation_metrics,
    )

    trainer._save_checkpoint(
        epoch=1,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        selection_metric="val_distortion",
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 1

    assert (
        checkpoint["selection_metric"]
        == "val_distortion"
    )

    assert (
        checkpoint["train_metrics"]
        == train_metrics
    )

    assert (
        checkpoint["validation_metrics"]
        == validation_metrics
    )

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "history" in checkpoint

def test_fit_plots_validation_latent_space_initially_and_each_epoch() -> None:
    model = LatentPlotModel()

    visualizer = RecordingLatentVisualizer()

    trainer = VQVAETrainer(
        model=model,
        objective=object(),
        settings=VQVAETrainerSettings(
            epochs=2,
            device="cpu",
            plot_validation_latent_space=True,
            restore_best_model=False,
        ),
        visualizer=visualizer,
    )

    train_loader = [
        torch.zeros(4, 2),
    ]

    validation_loader = [
        torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
                [0.4, 0.4],
            ]
        )
    ]

    trainer.train_epoch = lambda loader: {
        "loss": 1.0,
        "distortion": 1.0,
        "perplexity": 1.0,
        "utilization": 1.0,
        "entropy": 1.0,
        "dead_codewords": 0,
    }

    trainer.validate = lambda loader: {
        "loss": 1.0,
        "distortion": 1.0,
        "perplexity": 1.0,
        "utilization": 1.0,
        "entropy": 1.0,
        "dead_codewords": 0,
    }

    trainer.fit(
        train_loader,
        validation_loader,
    )

    assert len(visualizer.calls) == 3

    assert (
        visualizer.calls[0]["title"]
        == "Validation latent space — initial"
    )

    assert (
        visualizer.calls[1]["title"]
        == "Validation latent space — epoch 1"
    )

    assert (
        visualizer.calls[2]["title"]
        == "Validation latent space — epoch 2"
    )

def test_validation_latent_space_uses_all_validation_samples() -> None:
    model = LatentPlotModel()

    visualizer = RecordingLatentVisualizer()

    trainer = VQVAETrainer(
        model=model,
        objective=object(),
        settings=VQVAETrainerSettings(
            epochs=1,
            device="cpu",
            plot_validation_latent_space=True,
            restore_best_model=False,
        ),
        visualizer=visualizer,
    )

    validation_batch_1 = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
        ]
    )

    validation_batch_2 = torch.tensor(
        [
            [1.0, 1.0],
            [1.1, 1.1],
        ]
    )

    validation_loader = [
        validation_batch_1,
        validation_batch_2,
    ]

    trainer.train_epoch = lambda loader: {
        "loss": 1.0,
        "distortion": 1.0,
        "perplexity": 1.0,
        "utilization": 1.0,
        "entropy": 1.0,
        "dead_codewords": 0,
    }

    trainer.validate = lambda loader: {
        "loss": 1.0,
        "distortion": 1.0,
        "perplexity": 1.0,
        "utilization": 1.0,
        "entropy": 1.0,
        "dead_codewords": 0,
    }

    trainer.fit(
        train_loader=[],
        validation_loader=validation_loader,
    )

    # Initial plot + epoch 1 plot.
    assert len(visualizer.calls) == 2

    expected = torch.cat(
        [
            validation_batch_1,
            validation_batch_2,
        ],
        dim=0,
    ).numpy()

    for call in visualizer.calls:
        assert call["samples"].shape == (5, 2)
        assert np.array_equal(
            call["samples"],
            expected,
        )

def test_fit_does_not_plot_validation_latent_space_by_default() -> None:
    model = LatentPlotModel()

    visualizer = RecordingLatentVisualizer()

    trainer = VQVAETrainer(
        model=model,
        objective=object(),
        settings=VQVAETrainerSettings(
            epochs=2,
            device="cpu",
            restore_best_model=False,
        ),
        visualizer=visualizer,
    )

    trainer.train_epoch = lambda loader: {
        "loss": 1.0,
        "distortion": 1.0,
        "perplexity": 1.0,
        "utilization": 1.0,
        "entropy": 1.0,
        "dead_codewords": 0,
    }

    trainer.validate = lambda loader: {
        "loss": 1.0,
        "distortion": 1.0,
        "perplexity": 1.0,
        "utilization": 1.0,
        "entropy": 1.0,
        "dead_codewords": 0,
    }

    trainer.fit(
        train_loader=[],
        validation_loader=[
            torch.zeros(5, 2),
        ],
    )

    assert visualizer.calls == []

def test_plot_validation_latent_space_requires_visualizer() -> None:
    model = LatentPlotModel()

    with pytest.raises(
        ValueError,
        match="plot_validation_latent_space requires a Visualizer",
    ):
        VQVAETrainer(
            model=model,
            objective=object(),
            settings=VQVAETrainerSettings(
                epochs=1,
                device="cpu",
                plot_validation_latent_space=True,
            ),
            visualizer=None,
        )

def _make_visualization_test_trainer():
    trainer = VQVAETrainer.__new__(
        VQVAETrainer
    )

    trainer.device = "cpu"

    trainer.settings = SimpleNamespace(
        plot_validation_latent_space=True
    )

    trainer.model = SimpleNamespace(
        eval=Mock(),
        encode=lambda batch: batch[:, :2],
        quantizer=SimpleNamespace(
            settings=SimpleNamespace(
                embedding_dim=2
            ),
            embedding_mean=SimpleNamespace(
                weight=torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            ),
        ),
    )

    trainer.visualizer = SimpleNamespace(
        plot_latent_space=Mock()
    )

    trainer.validation_latent_figures = []

    return trainer

def test_collect_validation_latents_returns_aligned_targets():
    validation_data = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [3, 7, 2, 9],
        dtype=torch.long,
    )

    class ValidationDataset(
        torch.utils.data.Dataset
    ):
        def __len__(self):
            return len(validation_data)

        def __getitem__(self, index):
            return validation_data[index]

    dataset = ValidationDataset()
    dataset.targets = targets

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    trainer = _make_visualization_test_trainer()

    latents, collected_targets = (
        trainer._collect_validation_latents(
            loader
        )
    )

    assert torch.equal(
        latents,
        validation_data[:, :2],
    )

    assert torch.equal(
        collected_targets,
        targets,
    )

def test_plot_validation_latent_space_passes_targets():
    validation_data = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    class ValidationDataset(
        torch.utils.data.Dataset
    ):
        def __len__(self):
            return len(validation_data)

        def __getitem__(self, index):
            return validation_data[index]

    dataset = ValidationDataset()
    dataset.targets = targets

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    trainer = _make_visualization_test_trainer()

    trainer._plot_validation_latent_space(
        loader,
        epoch=None,
    )

    trainer.visualizer.plot_latent_space.assert_called_once()

    call = (
        trainer.visualizer
        .plot_latent_space.call_args
    )

    assert np.array_equal(
        call.kwargs["targets"],
        targets.numpy(),
    )

    assert np.array_equal(
        call.args[0],
        validation_data[:, :2].numpy(),
    )

    assert np.array_equal(
        call.kwargs["codewords"],
        trainer.model.quantizer.embedding_mean.weight.numpy(),
    )

    assert call.kwargs["title"] == (
        "Validation latent space — initial"
    )

def test_collect_validation_latents_allows_missing_targets():
    validation_data = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    class ValidationDataset(
        torch.utils.data.Dataset
    ):
        def __len__(self):
            return len(validation_data)

        def __getitem__(self, index):
            return validation_data[index]

    loader = torch.utils.data.DataLoader(
        ValidationDataset(),
        batch_size=1,
        shuffle=False,
    )

    trainer = _make_visualization_test_trainer()

    latents, targets = (
        trainer._collect_validation_latents(
            loader
        )
    )

    assert torch.equal(
        latents,
        validation_data[:, :2],
    )

    assert targets is None

