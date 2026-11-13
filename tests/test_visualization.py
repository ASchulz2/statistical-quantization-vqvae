"""
Tests for VQ-VAE visualization orchestration.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import numpy as np

import visualization as visualization_module
from settings import VQVAEExperimentSettings
from visualization import visualize_results
from matplotlib.figure import Figure

class RecordingVisualizer:
    """Visualization test double recording method calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.reconstruction_figure = Figure()
        self.reconstruction_residual_figure = Figure()
        self.generation_figure = Figure()
        self.codebook_figure = Figure()
        self.training_validation_figures = [
            Figure(),
            Figure(),
        ]
        self.latent_space_figure = Figure()

    def plot_training_validation_metrics(
        self,
        history,
    ) -> list[Figure]:
        self.calls.append(
            (
                "plot_training_validation_metrics",
                {
                    "history": history,
                },
            )
        )

        return self.training_validation_figures

    def plot_reconstructions(
        self,
        original,
        reconstructed,
        n_images,
        n_columns,
        title,
    ) -> Figure:
        self.calls.append(
            (
                "plot_reconstructions",
                {
                    "original": original,
                    "reconstructed": reconstructed,
                    "n_images": n_images,
                    "n_columns": n_columns,
                    "title": title,
                },
            )
        )

        return self.reconstruction_figure

    def plot_reconstruction_residuals(
        self,
        original,
        reconstructed,
        n_images,
        n_columns,
        title,
    ) -> Figure:
        self.calls.append(
            (
                "plot_reconstruction_residuals",
                {
                    "original": original,
                    "reconstructed": reconstructed,
                    "n_images": n_images,
                    "n_columns": n_columns,
                    "title": title,
                },
            )
        )

        return self.reconstruction_residual_figure

    def plot_generations(
        self,
        images,
        n_images,
        title,
    ) -> Figure:
        self.calls.append(
            (
                "plot_generations",
                {
                    "images": images,
                    "n_images": n_images,
                    "title": title,
                },
            )
        )

        return self.generation_figure

    def plot_codebook_images(
        self,
        decoder,
        codewords,
        device,
        n_columns,
        title,
    ) -> Figure:
        self.calls.append(
            (
                "plot_codebook_images",
                {
                    "decoder": decoder,
                    "codewords": codewords,
                    "device": device,
                    "n_columns": n_columns,
                    "title": title,
                },
            )
        )

        return self.codebook_figure

    def plot_latent_space(
        self,
        samples,
        codewords=None,
        targets=None,
        title="",
    ) -> Figure:
        self.calls.append(
            (
                "plot_latent_space",
                {
                    "samples": samples,
                    "codewords": codewords,
                    "targets": targets,
                    "title": title,
                },
            )
        )

        return self.latent_space_figure


class DummyHistory:
    """Minimal training history double."""

    def as_dict(self) -> dict:
        return {
            "loss": [1.0, 0.5],
            "val_loss": [1.1, 0.6],
        }


class DummyModel:
    """Minimal model used by visualization orchestration."""

    def __init__(self) -> None:
        self.decoder = object()

        self.quantizer = SimpleNamespace(
            embedding_mean=SimpleNamespace(
                weight=torch.zeros(
                    4,
                    2,
                )
            )
        )

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return z


def create_test_metrics() -> dict:
    """Create representative visualization inputs."""
    return {
        "originals": torch.rand(
            5,
            8,
        ),
        "reconstructions": torch.rand(
            5,
            8,
        ),
        "latents": torch.rand(
            5,
            2,
        ),
    }


def test_visualize_results_calls_expected_plots(
    monkeypatch,
) -> None:
    visualizer = RecordingVisualizer()
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=2,
        n_visualization_images=3,
        n_generative_images=2,
        plot_codebook=False,
        noise_scale=0.0,
    )

    def fake_sample_quantized_latents(
        model,
        z_e,
        noise_scale,
    ):
        return z_e

    monkeypatch.setattr(
        visualization_module,
        "sample_quantized_latents",
        fake_sample_quantized_latents,
    )

    figures = visualize_results(
        model=model,
        history=DummyHistory(),
        evaluation_metrics=create_test_metrics(),
        visualizer=visualizer,
        dataset_name="mnist",
        settings=settings,
        evaluation_split="test",
    )

    assert figures == [
        *visualizer.training_validation_figures,
        visualizer.reconstruction_figure,
        visualizer.reconstruction_residual_figure,
        visualizer.generation_figure,
        visualizer.latent_space_figure,
    ]

    metric_count = len(
        visualizer.training_validation_figures
    )

    assert figures[
        metric_count
    ] is visualizer.reconstruction_figure

    assert figures[
        metric_count + 1
    ] is visualizer.reconstruction_residual_figure

    assert figures[
        metric_count + 2
    ] is visualizer.generation_figure

    assert figures[
        metric_count + 3
    ] is visualizer.latent_space_figure

    call_names = [
        name
        for name, _ in visualizer.calls
    ]

    assert call_names == [
        "plot_training_validation_metrics",
        "plot_reconstructions",
        "plot_reconstruction_residuals",
        "plot_generations",
        "plot_latent_space",
    ]

    reconstruction_call = next(
        data
        for name, data in visualizer.calls
        if name == "plot_reconstructions"
    )

    assert reconstruction_call["n_images"] == 3
    assert reconstruction_call["n_columns"] == 8

    generation_call = next(
        data
        for name, data in visualizer.calls
        if name == "plot_generations"
    )

    assert generation_call["n_images"] == 2


def test_visualize_results_can_plot_codebook(
    monkeypatch,
) -> None:
    visualizer = RecordingVisualizer()
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=4,
        n_visualization_images=2,
        n_generative_images=2,
        plot_codebook=True,
    )

    monkeypatch.setattr(
        visualization_module,
        "sample_quantized_latents",
        lambda model, z_e, noise_scale: z_e,
    )

    figures = visualize_results(
        model=model,
        history=DummyHistory(),
        evaluation_metrics=create_test_metrics(),
        visualizer=visualizer,
        dataset_name="mnist",
        settings=settings,
        evaluation_split="test",
    )

    call_names = [
        name
        for name, _ in visualizer.calls
    ]

    assert "plot_codebook_images" in call_names
    assert "plot_latent_space" not in call_names

    assert figures == [
        *visualizer.training_validation_figures,
        visualizer.reconstruction_figure,
        visualizer.reconstruction_residual_figure,
        visualizer.generation_figure,
        visualizer.codebook_figure,
    ]

    assert figures[-1] is (
        visualizer.codebook_figure
    )


def test_visualize_results_skips_latent_space_when_not_two_dimensional(
    monkeypatch,
) -> None:
    visualizer = RecordingVisualizer()
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=8,
        n_visualization_images=2,
        n_generative_images=2,
        plot_codebook=False,
    )

    monkeypatch.setattr(
        visualization_module,
        "sample_quantized_latents",
        lambda model, z_e, noise_scale: z_e,
    )

    visualize_results(
        model=model,
        history=DummyHistory(),
        evaluation_metrics=create_test_metrics(),
        visualizer=visualizer,
        dataset_name="mnist",
        settings=settings,
        evaluation_split="test",
    )

    call_names = [
        name
        for name, _ in visualizer.calls
    ]

    assert "plot_latent_space" not in call_names

def test_visualize_results_uses_all_latent_samples(
    monkeypatch,
) -> None:
    visualizer = RecordingVisualizer()
    settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=2,
        n_visualization_images=3,
        n_generative_images=2,
        plot_codebook=False,
        noise_scale=0.0,
    )

    model = DummyModel()
    test_metrics = create_test_metrics()
    

    # Deliberately provide more latent samples than the
    # visualization image limit.
    latents = torch.arange(
        20,
        dtype=torch.float32,
    ).reshape(10, 2)

    test_metrics["latents"] = latents

    captured_latents: list[np.ndarray] = []

    def record_latent_space(
        samples,
        codewords=None,
        targets=None,
        title="",
    ):
        captured_latents.append(samples)

    monkeypatch.setattr(
        visualizer,
        "plot_latent_space",
        record_latent_space,
    )

    def fake_sample_quantized_latents(
        model,
        z_e,
        noise_scale,
    ):
        return z_e

    monkeypatch.setattr(
        visualization_module,
        "sample_quantized_latents",
        fake_sample_quantized_latents,
    )

    visualize_results(
        model=model,
        history=DummyHistory(),
        evaluation_metrics=test_metrics,
        visualizer=visualizer,
        dataset_name="mnist",
        settings=settings,
        evaluation_split="test",
    )

    assert len(captured_latents) == 1

    plotted_latents = captured_latents[0]

    assert plotted_latents.shape == (10, 2)

    np.testing.assert_array_equal(
        plotted_latents,
        latents.numpy(),
    )
