"""
Tests for stochastic VQ-VAE generation experiments.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import torch
from matplotlib.figure import Figure

import stochastic_generation as stochastic_module
from settings import VQVAEExperimentSettings

import pytest


class DummyLoader:
    """Minimal iterable loader for stochastic-generation tests."""

    def __init__(
        self,
        batches: list[torch.Tensor] | None = None,
    ) -> None:
        self.batches = (
            batches
            if batches is not None
            else [
                torch.zeros(
                    4,
                    4,
                )
            ]
        )

    def __iter__(self):
        yield from self.batches


class DummyEmbedding:
    """Minimal callable embedding table."""

    def __call__(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                indices.float(),
                indices.float() + 1.0,
            ],
            dim=-1,
        )


class DummyQuantizer:
    """Minimal statistical quantizer."""

    def __init__(
        self,
        use_variance: bool = True,
        has_logvar: bool = True,
    ) -> None:
        self.settings = Mock(
            use_variance=use_variance,
        )

        self.embedding_mean = DummyEmbedding()

        if has_logvar:

            self.embedding_logvar = (
                DummyEmbedding()
            )

    def assign(
        self,
        z_e: torch.Tensor,
    ) -> torch.Tensor:
        return torch.arange(
            z_e.shape[0],
            dtype=torch.long,
            device=z_e.device,
        )

    def quantize(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding_mean(
            indices
        )


class DummyModel:
    """Minimal model for stochastic-generation tests."""

    def __init__(
        self,
        use_variance: bool = True,
        has_logvar: bool = True,
    ) -> None:
        self.quantizer = DummyQuantizer(
            use_variance=use_variance,
            has_logvar=has_logvar,
        )

        self.eval_called = False
        self.to_device = None
        self.loaded_state = None

    def load_state_dict(
        self,
        state_dict,
    ) -> None:
        self.loaded_state = state_dict

    def to(
        self,
        device,
    ):
        self.to_device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def encode(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return observations[
            :,
            :2,
        ]

    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        return latents


def create_generation_settings(
    **kwargs,
) -> stochastic_module.StochasticGenerationSettings:
    """Create small valid stochastic-generation settings."""
    defaults = {
        "n_uncertainty_observations": 2,
        "n_conditional_observations": 2,
        "n_conditional_samples": 2,
        "n_columns": 2,
        "alphas": (
            0.0,
            1.0,
        ),
        "conditional_alpha": 1.0,
    }

    defaults.update(
        kwargs
    )

    return (
        stochastic_module.StochasticGenerationSettings(
            **defaults
        )
    )


def test_stochastic_generation_settings_accepts_valid_values() -> None:
    settings = create_generation_settings()

    assert settings.n_uncertainty_observations == 2
    assert settings.n_conditional_observations == 2
    assert settings.n_conditional_samples == 2
    assert settings.n_columns == 2
    assert settings.alphas == (
        0.0,
        1.0,
    )
    assert settings.conditional_alpha == 1.0


def test_stochastic_generation_settings_rejects_non_positive_observation_count() -> None:
    try:
        create_generation_settings(
            n_uncertainty_observations=0
        )
    except ValueError as exc:
        assert str(exc) == (
            "n_uncertainty_observations must be positive."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_stochastic_generation_settings_rejects_empty_alphas() -> None:
    try:
        create_generation_settings(
            alphas=()
        )
    except ValueError as exc:
        assert str(exc) == (
            "alphas must not be empty."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_stochastic_generation_settings_rejects_negative_alpha() -> None:
    try:
        create_generation_settings(
            alphas=(-1.0,)
        )
    except ValueError as exc:
        assert str(exc) == (
            "All alpha values must be non-negative."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_stochastic_generation_settings_rejects_negative_conditional_alpha() -> None:
    try:
        create_generation_settings(
            conditional_alpha=-1.0
        )
    except ValueError as exc:
        assert str(exc) == (
            "conditional_alpha must be non-negative."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_checkpoint_path() -> None:
    settings = VQVAEExperimentSettings(
        checkpoint_directory=Path(
            "checkpoints"
        )
    )

    result = stochastic_module.checkpoint_path(
        settings,
        "mnist",
    )

    assert result == Path(
        "checkpoints/vqvae_mnist.pt"
    )


def test_select_evaluation_loader_returns_validation_loader() -> None:
    train_loader = object()
    validation_loader = object()
    test_loader = object()

    result = (
        stochastic_module.select_evaluation_loader(
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            evaluation_split="validation",
        )
    )

    assert result is validation_loader


def test_select_evaluation_loader_returns_test_loader() -> None:
    train_loader = object()
    validation_loader = object()
    test_loader = object()

    result = (
        stochastic_module.select_evaluation_loader(
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            evaluation_split="test",
        )
    )

    assert result is test_loader


def test_select_evaluation_loader_rejects_unsupported_split() -> None:
    try:
        stochastic_module.select_evaluation_loader(
            train_loader=object(),
            validation_loader=object(),
            test_loader=object(),
            evaluation_split="train",
        )
    except ValueError as exc:
        assert str(exc) == (
            "evaluation_split must be "
            "'validation' or 'test'."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_collect_evaluation_examples_collects_requested_count() -> None:
    loader = DummyLoader(
        [
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            torch.tensor(
                [
                    [5.0, 6.0],
                    [7.0, 8.0],
                ]
            ),
        ]
    )

    result = (
        stochastic_module.collect_evaluation_examples(
            loader,
            3,
        )
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    assert torch.equal(
        result,
        expected,
    )


def test_collect_evaluation_examples_rejects_non_positive_count() -> None:
    try:
        stochastic_module.collect_evaluation_examples(
            DummyLoader(),
            0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "n_examples must be positive."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_collect_evaluation_examples_rejects_non_tensor_batch() -> None:
    loader = DummyLoader(
        [
            "invalid"
        ]
    )

    try:
        stochastic_module.collect_evaluation_examples(
            loader,
            1,
        )
    except TypeError as exc:
        assert str(exc) == (
            "Expected the evaluation loader to return tensors."
        )
    else:
        raise AssertionError(
            "Expected TypeError."
        )


def test_collect_evaluation_examples_rejects_empty_loader() -> None:
    try:
        stochastic_module.collect_evaluation_examples(
            DummyLoader(
                []
            ),
            1,
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "No evaluation examples were collected."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_load_model_restores_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path
        /
        "model.pt"
    )

    torch.save(
        {
            "model_state_dict": {
                "weight": 1,
            }
        },
        checkpoint_path,
    )

    expected_model = DummyModel()

    monkeypatch.setattr(
        stochastic_module,
        "create_model",
        Mock(
            return_value=expected_model
        ),
    )

    settings = VQVAEExperimentSettings(
        device="cpu"
    )

    result = stochastic_module.load_model(
        checkpoint=checkpoint_path,
        input_dim=4,
        settings=settings,
    )

    assert result is expected_model
    assert expected_model.loaded_state == {
        "weight": 1,
    }
    assert expected_model.to_device == "cpu"
    assert expected_model.eval_called is True


def test_load_model_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path
        /
        "missing.pt"
    )

    try:
        stochastic_module.load_model(
            checkpoint=checkpoint_path,
            input_dim=4,
            settings=VQVAEExperimentSettings(
                device="cpu"
            ),
        )
    except FileNotFoundError as exc:
        assert str(exc) == (
            f"Checkpoint not found: {checkpoint_path}"
        )
    else:
        raise AssertionError(
            "Expected FileNotFoundError."
        )


def test_load_model_rejects_invalid_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path
        /
        "invalid.pt"
    )

    torch.save(
        [],
        checkpoint_path,
    )

    try:
        stochastic_module.load_model(
            checkpoint=checkpoint_path,
            input_dim=4,
            settings=VQVAEExperimentSettings(
                device="cpu"
            ),
        )
    except TypeError as exc:
        assert str(exc) == (
            "Expected a dictionary checkpoint."
        )
    else:
        raise AssertionError(
            "Expected TypeError."
        )


def test_load_model_rejects_missing_state_dict(
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path
        /
        "invalid.pt"
    )

    torch.save(
        {},
        checkpoint_path,
    )

    with pytest.raises(
        KeyError,
        match="Checkpoint does not contain 'model_state_dict'\\.",
    ):
        stochastic_module.load_model(
            checkpoint=checkpoint_path,
            input_dim=4,
            settings=VQVAEExperimentSettings(
                device="cpu"
            ),
        )


def test_conditional_means_returns_codeword_means() -> None:
    model = DummyModel()

    indices = torch.tensor(
        [
            1,
            2,
        ],
        dtype=torch.long,
    )

    result = (
        stochastic_module.conditional_means(
            model,
            indices,
        )
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )

    assert torch.equal(
        result,
        expected,
    )


def test_conditional_stds_returns_positive_standard_deviations() -> None:
    model = DummyModel(
        use_variance=True,
        has_logvar=True,
    )

    indices = torch.tensor(
        [
            1,
            2,
        ],
        dtype=torch.long,
    )

    result = (
        stochastic_module.conditional_stds(
            model,
            indices,
        )
    )

    assert result.shape == (
        2,
        2,
    )

    assert torch.all(
        result > 0.0
    )


def test_conditional_stds_rejects_disabled_variance() -> None:
    model = DummyModel(
        use_variance=False,
        has_logvar=True,
    )

    indices = torch.tensor(
        [1],
        dtype=torch.long,
    )

    try:
        stochastic_module.conditional_stds(
            model,
            indices,
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "Variance sampling is disabled in the loaded model."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_conditional_stds_rejects_missing_logvar() -> None:
    model = DummyModel(
        use_variance=True,
        has_logvar=False,
    )

    indices = torch.tensor(
        [1],
        dtype=torch.long,
    )

    try:
        stochastic_module.conditional_stds(
            model,
            indices,
        )
    except AttributeError as exc:
        assert str(exc) == (
            "The loaded quantizer has no embedding_logvar."
        )
    else:
        raise AssertionError(
            "Expected AttributeError."
        )


def test_plot_uncertainty_comparison_creates_figure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        stochastic_module,
        "prepare_images",
        lambda tensor: [
            torch.zeros(
                2,
                2,
            ).numpy()
            for _ in range(
                tensor.shape[0]
            )
        ],
    )

    monkeypatch.setattr(
        stochastic_module.plt,
        "show",
        Mock(),
    )

    generated = {
        0.0: torch.zeros(
            2,
            4,
        ),
        1.0: torch.ones(
            2,
            4,
        ),
    }

    stochastic_module.plot_uncertainty_comparison(
        generated,
        n_columns=2,
        title="Test",
    )

    assert (
        stochastic_module.plt.show.called
    )

    for figure_number in (
        stochastic_module.plt.get_fignums()
    ):
        stochastic_module.plt.close(
            figure_number
        )


def test_plot_uncertainty_comparison_rejects_empty_input() -> None:
    try:
        stochastic_module.plot_uncertainty_comparison(
            {},
            title="Test",
        )
    except ValueError as exc:
        assert str(exc) == (
            "No generated samples were supplied."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_plot_conditional_generations_creates_figure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        stochastic_module,
        "prepare_images",
        lambda tensor: [
            torch.zeros(
                2,
                2,
            ).numpy()
            for _ in range(
                tensor.shape[0]
            )
        ],
    )

    monkeypatch.setattr(
        stochastic_module.plt,
        "show",
        Mock(),
    )

    observations = torch.zeros(
        2,
        4,
    )

    generations = torch.zeros(
        3,
        2,
        4,
    )

    indices = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    stochastic_module.plot_conditional_generations(
        observations,
        generations,
        indices,
        title="Test",
    )

    assert (
        stochastic_module.plt.show.called
    )

    for figure_number in (
        stochastic_module.plt.get_fignums()
    ):
        stochastic_module.plt.close(
            figure_number
        )


def test_plot_conditional_generations_rejects_wrong_observation_shape() -> None:
    observations = torch.zeros(
        2,
        4,
        1,
    )

    generations = torch.zeros(
        3,
        2,
        4,
    )

    indices = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    try:
        stochastic_module.plot_conditional_generations(
            observations,
            generations,
            indices,
            title="Test",
        )
    except ValueError as exc:
        assert str(exc) == (
            "observations must have shape "
            "(n_observations, input_dim)."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_plot_conditional_generations_rejects_wrong_generation_shape() -> None:
    observations = torch.zeros(
        2,
        4,
    )

    generations = torch.zeros(
        2,
        4,
    )

    indices = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    try:
        stochastic_module.plot_conditional_generations(
            observations,
            generations,
            indices,
            title="Test",
        )
    except ValueError as exc:
        assert str(exc) == (
            "generations must have shape "
            "(n_samples, n_observations, input_dim)."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_plot_conditional_generations_rejects_mismatched_indices() -> None:
    observations = torch.zeros(
        2,
        4,
    )

    generations = torch.zeros(
        3,
        2,
        4,
    )

    indices = torch.tensor(
        [0],
        dtype=torch.long,
    )

    try:
        stochastic_module.plot_conditional_generations(
            observations,
            generations,
            indices,
            title="Test",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Observation and index counts do not match."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_plot_conditional_generations_rejects_mismatched_generation_count() -> None:
    observations = torch.zeros(
        2,
        4,
    )

    generations = torch.zeros(
        3,
        3,
        4,
    )

    indices = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    try:
        stochastic_module.plot_conditional_generations(
            observations,
            generations,
            indices,
            title="Test",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Observation and generation counts do not match."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_run_experiment_uses_requested_validation_split(
    monkeypatch,
) -> None:
    vqvae_settings = VQVAEExperimentSettings(
        device="cpu",
        use_cifar10=False,
    )

    generation_settings = create_generation_settings()

    train_loader = DummyLoader(
        [
            torch.zeros(
                4,
                4,
            )
        ]
    )

    validation_loader = DummyLoader(
        [
            torch.arange(
                16,
                dtype=torch.float32,
            ).reshape(
                4,
                4,
            )
        ]
    )

    test_loader = DummyLoader(
        [
            torch.ones(
                4,
                4,
            )
        ]
    )

    expected_model = DummyModel()

    checkpoint = Path(
        "checkpoints/vqvae_mnist.pt"
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stochastic_module,
        "prepare_dataset",
        Mock(
            return_value=(
                "mnist",
                train_loader,
                validation_loader,
                test_loader,
            )
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "checkpoint_path",
        Mock(
            return_value=checkpoint
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "load_model",
        Mock(
            return_value=expected_model
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "collect_evaluation_examples",
        lambda loader, n_examples: (
            captured.update(
                loader=loader,
                n_examples=n_examples,
            )
            or validation_loader.batches[0]
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "conditional_means",
        lambda model, indices: torch.zeros(
            len(indices),
            2,
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "conditional_stds",
        lambda model, indices: torch.ones(
            len(indices),
            2,
        ),
    )

    def fake_sample_codeword_latents(
        model,
        indices,
        noise_scale,
        n_samples,
    ):
        return torch.zeros(
            n_samples,
            len(indices),
            2,
        )

    monkeypatch.setattr(
        stochastic_module,
        "sample_codeword_latents",
        fake_sample_codeword_latents,
    )

    uncertainty_plot = Mock()
    conditional_plot = Mock()

    monkeypatch.setattr(
        stochastic_module,
        "plot_uncertainty_comparison",
        uncertainty_plot,
    )

    monkeypatch.setattr(
        stochastic_module,
        "plot_conditional_generations",
        conditional_plot,
    )

    stochastic_module.run_experiment(
        vqvae_settings=vqvae_settings,
        generation_settings=generation_settings,
        evaluation_split="validation",
    )

    assert captured["loader"] is (
        validation_loader
    )

    assert captured["n_examples"] == 2

    uncertainty_plot.assert_called_once()

    conditional_plot.assert_called_once()

    assert (
        "validation"
        in uncertainty_plot.call_args.kwargs["title"]
    )

    assert (
        "validation"
        in conditional_plot.call_args.kwargs["title"]
    )


def test_run_experiment_uses_default_test_split(
    monkeypatch,
) -> None:
    vqvae_settings = VQVAEExperimentSettings(
        device="cpu",
    )

    generation_settings = create_generation_settings()

    train_loader = DummyLoader()
    validation_loader = DummyLoader(
        [
            torch.zeros(
                4,
                4,
            )
        ]
    )
    test_loader = DummyLoader(
        [
            torch.ones(
                4,
                4,
            )
        ]
    )

    expected_model = DummyModel()

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stochastic_module,
        "prepare_dataset",
        Mock(
            return_value=(
                "mnist",
                train_loader,
                validation_loader,
                test_loader,
            )
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "checkpoint_path",
        Mock(
            return_value=Path(
                "checkpoints/vqvae_mnist.pt"
            )
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "load_model",
        Mock(
            return_value=expected_model
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "collect_evaluation_examples",
        lambda loader, n_examples: (
            captured.update(
                loader=loader,
                n_examples=n_examples,
            )
            or test_loader.batches[0]
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "conditional_means",
        lambda model, indices: torch.zeros(
            len(indices),
            2,
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "conditional_stds",
        lambda model, indices: torch.ones(
            len(indices),
            2,
        ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "sample_codeword_latents",
        lambda model, indices, noise_scale, n_samples:
            torch.zeros(
                n_samples,
                len(indices),
                2,
            ),
    )

    monkeypatch.setattr(
        stochastic_module,
        "plot_uncertainty_comparison",
        Mock(),
    )

    monkeypatch.setattr(
        stochastic_module,
        "plot_conditional_generations",
        Mock(),
    )

    stochastic_module.run_experiment(
        vqvae_settings=vqvae_settings,
        generation_settings=generation_settings,
    )

    assert captured["loader"] is (
        test_loader
    )
