"""
Tests for the Visualizer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from classes.visualizer import (
    VisualizationSettings,
    Visualizer,
)
from unittest.mock import Mock

from matplotlib.figure import Figure

@pytest.fixture
def visualizer() -> Visualizer:
    return Visualizer(
        VisualizationSettings(
            figsize=(4, 4),
            dpi=80,
        )
    )


@pytest.fixture(autouse=True)
def disable_show(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        plt,
        "show",
        lambda: plt.close("all"),
    )


def test_visualization_settings() -> None:
    settings = VisualizationSettings()

    assert settings.figsize == (
        8,
        6,
    )

    assert settings.dpi == 100


@pytest.mark.parametrize(
    "figsize",
    [
        (0, 6),
        (8, 0),
        (-1, 6),
    ],
)
def test_invalid_figsize_raises(
    figsize: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        VisualizationSettings(
            figsize=figsize
        )


def test_invalid_dpi_raises() -> None:
    with pytest.raises(ValueError):
        VisualizationSettings(
            dpi=0
        )


def test_latent_space(
    visualizer: Visualizer,
) -> None:
    samples = np.random.rand(
        10,
        2,
    )

    codewords = np.random.rand(
        4,
        2,
    )

    visualizer.plot_latent_space(
        samples,
        codewords,
    )


def test_latent_space_rejects_non_2d() -> None:
    visualizer = Visualizer(
        VisualizationSettings()
    )

    samples = np.random.rand(
        10,
        3,
    )

    with pytest.raises(ValueError):
        visualizer.plot_latent_space(
            samples
        )


def test_sequence(
    visualizer: Visualizer,
) -> None:
    original = np.random.rand(
        20,
        3,
    )

    reconstructed = np.random.rand(
        20,
        3,
    )

    visualizer.plot_sequence(
        original,
        reconstructed,
        dimension=1,
    )


def test_autocorrelation(
    visualizer: Visualizer,
) -> None:
    visualizer.plot_autocorrelation(
        np.random.rand(10)
    )


def test_rate_distortion(
    visualizer: Visualizer,
) -> None:
    visualizer.plot_rate_distortion(
        rates=[1.0, 2.0, 3.0],
        distortions=[
            0.5,
            0.3,
            0.2,
        ],
    )


def test_rate_distortion_rejects_length_mismatch() -> None:
    visualizer = Visualizer(
        VisualizationSettings()
    )

    with pytest.raises(ValueError):
        visualizer.plot_rate_distortion(
            rates=[1.0, 2.0],
            distortions=[0.5],
        )


def test_rate_distortion_comparison(
    visualizer: Visualizer,
) -> None:
    visualizer.plot_rate_distortion_comparison(
        direct=(
            [1.0, 2.0],
            [0.5, 0.3],
        ),
        predictive=(
            [1.0, 2.0],
            [0.6, 0.4],
        ),
    )


def test_quantization_cells(
    visualizer: Visualizer,
) -> None:
    samples = np.random.rand(
        10,
        2,
    )

    indices = np.zeros(
        10,
        dtype=int,
    )

    codewords = np.random.rand(
        2,
        2,
    )

    visualizer.plot_quantization_cells(
        samples,
        indices,
        codewords,
    )


def test_extract_rate_distortion(
    visualizer: Visualizer,
) -> None:
    results = [
        {
            "rate_bits": 1.0,
            "distortion": 0.5,
        },
        {
            "rate_bits": 2.0,
            "distortion": 0.25,
        },
    ]

    rates, distortions = (
        visualizer.extract_rate_distortion(
            results
        )
    )

    assert rates == [
        1.0,
        2.0,
    ]

    assert distortions == [
        0.5,
        0.25,
    ]


def test_plot_reconstructions(
    visualizer: Visualizer,
) -> None:
    original = np.random.rand(
        4,
        784,
    )

    reconstructed = np.random.rand(
        4,
        784,
    )

    visualizer.plot_reconstructions(
        original,
        reconstructed,
        n_images=4,
        n_columns=2,
    )


def test_plot_reconstruction_residuals(
    visualizer: Visualizer,
) -> None:
    original = np.random.rand(
        4,
        784,
    )

    reconstructed = np.random.rand(
        4,
        784,
    )

    visualizer.plot_reconstruction_residuals(
        original,
        reconstructed,
        n_images=4,
        n_columns=2,
    )


def test_plot_generations(
    visualizer: Visualizer,
) -> None:
    images = np.random.rand(
        4,
        784,
    )

    visualizer.plot_generations(
        images,
        n_images=4,
        n_columns=2,
    )


def test_plot_metrics(
    visualizer: Visualizer,
) -> None:
    history = {
        "loss": [
            2.0,
            1.0,
        ],
        "distortion": [
            1.0,
            0.5,
        ],
    }

    visualizer.plot_metrics(
        history
    )


def test_plot_training_validation_metrics(
    visualizer: Visualizer,
) -> None:
    history = {
        "loss": [
            2.0,
            1.0,
        ],
        "val_loss": [
            2.2,
            1.2,
        ],
    }

    visualizer.plot_training_validation_metrics(
        history,
        metrics=("loss",),
    )


def test_training_validation_length_mismatch_raises(
    visualizer: Visualizer,
) -> None:
    history = {
        "loss": [
            2.0,
            1.0,
        ],
        "val_loss": [
            2.2,
        ],
    }

    with pytest.raises(ValueError):
        visualizer.plot_training_validation_metrics(
            history,
            metrics=("loss",),
        )


def test_plot_codebook_images(
    visualizer: Visualizer,
) -> None:
    class DummyDecoder(nn.Module):

        def forward(
            self,
            codewords: torch.Tensor,
        ) -> torch.Tensor:
            return torch.zeros(
                len(codewords),
                784,
                device=codewords.device,
            )

    decoder = DummyDecoder()

    codewords = torch.zeros(
        4,
        2,
    )

    visualizer.plot_codebook_images(
        decoder=decoder,
        codewords=codewords,
        device="cpu",
        n_columns=2,
    )

    assert decoder.training is True


def test_plot_codebook_images_rejects_invalid_codewords(
    visualizer: Visualizer,
) -> None:
    with pytest.raises(ValueError):
        visualizer.plot_codebook_images(
            decoder=nn.Linear(2, 784),
            codewords=torch.zeros(
                2,
                2,
                1,
            ),
        )


def test_plot_codebook_images_rejects_invalid_columns(
    visualizer: Visualizer,
) -> None:
    with pytest.raises(ValueError):
        visualizer.plot_codebook_images(
            decoder=nn.Linear(2, 784),
            codewords=torch.zeros(
                2,
                2,
            ),
            n_columns=0,
        )


def test_recursive_residuals(
    visualizer: Visualizer,
) -> None:
    original = np.random.rand(
        2,
        784,
    )

    reconstructions = [
        np.random.rand(
            2,
            784,
        ),
        np.random.rand(
            2,
            784,
        ),
    ]

    visualizer.plot_recursive_residuals(
        original,
        reconstructions,
        n_images=2,
    )


def test_recursive_reconstructions(
    visualizer: Visualizer,
) -> None:
    original = np.random.rand(
        2,
        784,
    )

    reconstructions = [
        np.random.rand(
            2,
            784,
        ),
        np.random.rand(
            2,
            784,
        ),
    ]

    visualizer.plot_recursive_reconstructions(
        original,
        reconstructions,
        n_images=2,
    )


def test_prepare_images_wrapper(
    visualizer: Visualizer,
) -> None:
    images = torch.zeros(
        2,
        784,
    )

    prepared = visualizer._prepare_images(
        images
    )

    assert prepared.shape == (
        2,
        28,
        28,
    )


def test_flattened_to_image_wrapper(
    visualizer: Visualizer,
) -> None:
    images = np.zeros(
        (2, 784)
    )

    prepared = (
        visualizer._flattened_to_image(
            images
        )
    )

    assert prepared.shape == (
        2,
        28,
        28,
    )

def test_latent_space_with_targets(
    visualizer: Visualizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1],
        ],
        dtype=np.float32,
    )

    targets = np.array(
        [0, 0, 1, 1],
    )

    scatter_calls = []

    monkeypatch.setattr(
        plt,
        "scatter",
        lambda *args, **kwargs: scatter_calls.append(
            (args, kwargs)
        ),
    )

    monkeypatch.setattr(
        plt,
        "show",
        lambda: None,
    )

    visualizer.plot_latent_space(
        samples,
        targets=targets,
    )

    assert len(scatter_calls) == 2

    assert scatter_calls[0][0][0].shape == (2,)
    assert scatter_calls[1][0][0].shape == (2,)

    assert scatter_calls[0][1]["label"] == "0"
    assert scatter_calls[1][1]["label"] == "1"

    plt.close("all")

def test_latent_space_rejects_target_count_mismatch(
    visualizer: Visualizer,
) -> None:
    samples = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    targets = np.array(
        [0, 1, 2],
    )

    with pytest.raises(
        ValueError,
        match="Number of targets must match",
    ):
        visualizer.plot_latent_space(
            samples,
            targets=targets,
        )

def test_latent_space_rejects_non_1d_targets(
    visualizer: Visualizer,
) -> None:
    samples = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    targets = np.zeros(
        (4, 1),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="Targets must have shape",
    ):
        visualizer.plot_latent_space(
            samples,
            targets=targets,
        )

def test_latent_space_with_targets_does_not_require_legend_artists(
    visualizer,
    monkeypatch,
) -> None:
    samples = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1],
        ],
        dtype=np.float32,
    )

    targets = np.array([0, 0, 1, 1])

    legend_calls: list[tuple] = []

    monkeypatch.setattr(
        plt,
        "scatter",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        plt,
        "legend",
        lambda *args, **kwargs: legend_calls.append(
            (args, kwargs)
        ),
    )

    monkeypatch.setattr(
        plt,
        "show",
        lambda: None,
    )

    visualizer.plot_latent_space(
        samples,
        targets=targets,
    )

    assert legend_calls == []

def test_latent_space_calls_legend_when_labeled_artists_exist(
    visualizer,
    monkeypatch,
) -> None:
    samples = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    legend_calls: list[tuple] = []

    monkeypatch.setattr(
        plt,
        "legend",
        lambda *args, **kwargs: legend_calls.append(
            (args, kwargs)
        ),
    )

    monkeypatch.setattr(
        plt,
        "show",
        lambda: None,
    )

    visualizer.plot_latent_space(
        samples,
        targets=np.array([0, 0]),
    )

    assert len(legend_calls) == 1

def test_plot_latent_space_returns_figure():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    samples = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    figure = visualizer.plot_latent_space(
        samples
    )

    assert isinstance(
        figure,
        Figure,
    )

def test_plot_latent_space_does_not_show_figure(
    monkeypatch,
):
    visualizer = Visualizer(
        VisualizationSettings()
    )

    samples = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    show = Mock()

    monkeypatch.setattr(
        plt,
        "show",
        show,
    )

    visualizer.plot_latent_space(
        samples
    )

    show.assert_not_called()

def test_plot_images_returns_figure():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    images = torch.rand(
        16,
        784,
    )

    figure = visualizer.plot_images(
        images=images,
        n_images=16,
        n_columns=8,
        title="Training-set examples",
    )

    assert isinstance(
        figure,
        Figure,
    )

def test_plot_images_does_not_show_figure(
    monkeypatch,
):
    visualizer = Visualizer(
        VisualizationSettings()
    )

    images = torch.rand(
        16,
        784,
    )

    show = Mock()

    monkeypatch.setattr(
        plt,
        "show",
        show,
    )

    visualizer.plot_images(
        images=images,
        n_images=16,
        n_columns=8,
        title="Training-set examples",
    )

    show.assert_not_called()

def test_plot_reconstructions_returns_figure():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    original = torch.rand(
        16,
        784,
    )

    reconstructed = torch.rand(
        16,
        784,
    )

    figure = visualizer.plot_reconstructions(
        original=original,
        reconstructed=reconstructed,
        n_images=16,
        n_columns=8,
        title="Reconstructions",
    )

    assert isinstance(
        figure,
        Figure,
    )

def test_plot_reconstructions_does_not_show_figure(
    monkeypatch,
):
    visualizer = Visualizer(
        VisualizationSettings()
    )

    original = torch.rand(
        16,
        784,
    )

    reconstructed = torch.rand(
        16,
        784,
    )

    show = Mock()

    monkeypatch.setattr(
        plt,
        "show",
        show,
    )

    visualizer.plot_reconstructions(
        original=original,
        reconstructed=reconstructed,
        n_images=16,
        n_columns=8,
        title="Reconstructions",
    )

    show.assert_not_called()


def test_plot_generations_returns_figure():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    images = torch.rand(
        16,
        784,
    )

    figure = visualizer.plot_generations(
        images=images,
        n_images=16,
        n_columns=8,
        title="Generated samples",
    )

    assert isinstance(
        figure,
        Figure,
    )

def test_plot_generations_does_not_show_figure(
    monkeypatch,
):
    visualizer = Visualizer(
        VisualizationSettings()
    )

    images = torch.rand(
        16,
        784,
    )

    show = Mock()

    monkeypatch.setattr(
        plt,
        "show",
        show,
    )

    visualizer.plot_generations(
        images=images,
        n_images=16,
        n_columns=8,
        title="Generated samples",
    )

    show.assert_not_called()

def test_plot_codebook_images_returns_figure():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    decoder = Mock()

    decoder.training = False

    decoder.return_value = torch.rand(
        4,
        784,
    )

    codewords = torch.rand(
        4,
        2,
    )

    figure = visualizer.plot_codebook_images(
        decoder=decoder,
        codewords=codewords,
        device="cpu",
        n_columns=2,
        title="Decoded codebook",
    )

    assert isinstance(
        figure,
        Figure,
    )

def test_plot_codebook_images_does_not_show_figure(
    monkeypatch,
):
    visualizer = Visualizer(
        VisualizationSettings()
    )

    decoder = Mock()

    decoder.training = False

    decoder.return_value = torch.rand(
        4,
        784,
    )

    codewords = torch.rand(
        4,
        2,
    )

    show = Mock()

    monkeypatch.setattr(
        plt,
        "show",
        show,
    )

    visualizer.plot_codebook_images(
        decoder=decoder,
        codewords=codewords,
        device="cpu",
        n_columns=2,
        title="Decoded codebook",
    )

    show.assert_not_called()

def test_plot_training_validation_metrics_returns_figures():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    history = {
        "loss": [1.0, 0.8],
        "val_loss": [1.1, 0.9],
        "distortion": [0.5, 0.4],
        "val_distortion": [0.6, 0.5],
    }

    figures = (
        visualizer.plot_training_validation_metrics(
            history
        )
    )

    assert len(figures) == 2

    for figure in figures:
        assert isinstance(
            figure,
            Figure,
        )

def test_plot_training_validation_metrics_does_not_show_figures(
    monkeypatch,
):
    visualizer = Visualizer(
        VisualizationSettings()
    )

    history = {
        "loss": [1.0, 0.8],
        "val_loss": [1.1, 0.9],
    }

    show = Mock()

    monkeypatch.setattr(
        plt,
        "show",
        show,
    )

    figures = (
        visualizer.plot_training_validation_metrics(
            history
        )
    )

    assert len(figures) == 1

    show.assert_not_called()

def test_plot_training_validation_metrics_skips_missing_series():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    history = {
        "loss": [1.0, 0.8],
    }

    figures = (
        visualizer.plot_training_validation_metrics(
            history
        )
    )

    assert figures == []

def test_plot_metrics_returns_figures():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    history = {
        "loss": [1.0, 0.8],
        "perplexity": [2.0, 2.5],
    }

    figures = visualizer.plot_metrics(
        history
    )

    assert len(figures) == 2

    for figure in figures:
        assert isinstance(
            figure,
            Figure,
        )

def test_plot_metrics_does_not_show_figures(
    monkeypatch,
):
    visualizer = Visualizer(
        VisualizationSettings()
    )

    history = {
        "loss": [1.0, 0.8],
    }

    show = Mock()

    monkeypatch.setattr(
        plt,
        "show",
        show,
    )

    figures = visualizer.plot_metrics(
        history
    )

    assert len(figures) == 1

    show.assert_not_called()


def test_plot_metrics_skips_invalid_entries():
    visualizer = Visualizer(
        VisualizationSettings()
    )

    history = {
        "loss": [1.0, 0.8],
        "empty": [],
        "not_a_list": np.array([1.0]),
    }

    figures = visualizer.plot_metrics(
        history
    )

    assert len(figures) == 1