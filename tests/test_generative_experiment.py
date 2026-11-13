from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import generative_experiment
from settings import VQVAEExperimentSettings

from matplotlib.figure import Figure
from unittest.mock import Mock

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def experiment_settings(
    tmp_path: Path,
) -> generative_experiment.GenerativeExperimentSettings:
    return generative_experiment.GenerativeExperimentSettings(
        seed=42,
        device="cpu",
        use_cifar10=False,
        data_root="./data",
        checkpoint_directory=tmp_path,
        num_samples=4,
        noise_scale=0.5,
        batch_size=2,
        interactive=False,
    )




# ============================================================
# GenerativeExperimentSettings
# ============================================================


def test_generative_experiment_settings_defaults() -> None:
    settings = (
        generative_experiment.GenerativeExperimentSettings()
    )

    assert settings.seed == 42
    assert settings.use_cifar10 is False
    assert settings.num_samples == 32
    assert settings.noise_scale == 0.5
    assert settings.batch_size == 128
    assert settings.interactive is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "invalid"},
        {"num_samples": 0},
        {"noise_scale": -1.0},
        {"batch_size": 0},
    ],
)

def test_generative_experiment_settings_reject_invalid_values(
    kwargs: dict,
) -> None:
    with pytest.raises(ValueError):
        generative_experiment.GenerativeExperimentSettings(
            **kwargs
        )


# ============================================================
# Checkpoint path
# ============================================================


def test_checkpoint_path_for_mnist(
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
) -> None:
    path = generative_experiment.checkpoint_path(
        experiment_settings
    )

    assert path == (
        experiment_settings.checkpoint_directory
        / "vqvae_mnist.pt"
    )


def test_checkpoint_path_for_cifar10(
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
) -> None:
    settings = (
        generative_experiment.GenerativeExperimentSettings(
            checkpoint_directory=(
                experiment_settings.checkpoint_directory
            ),
            use_cifar10=True,
        )
    )

    path = generative_experiment.checkpoint_path(
        settings
    )

    assert path == (
        experiment_settings.checkpoint_directory
        / "vqvae_cifar10.pt"
    )


# ============================================================
# load_checkpoint
# ============================================================


def test_load_checkpoint_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        generative_experiment.load_checkpoint(
            tmp_path / "missing.pt",
            "cpu",
        )


def test_load_checkpoint_rejects_non_dictionary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.pt"

    torch.save(
        [1, 2, 3],
        path,
    )

    with pytest.raises(TypeError):
        generative_experiment.load_checkpoint(
            path,
            "cpu",
        )


def test_load_checkpoint_requires_model_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.pt"

    torch.save(
        {"settings": {}},
        path,
    )

    with pytest.raises(KeyError):
        generative_experiment.load_checkpoint(
            path,
            "cpu",
        )


def test_load_checkpoint_requires_settings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.pt"

    torch.save(
        {"model_state_dict": {}},
        path,
    )

    with pytest.raises(KeyError):
        generative_experiment.load_checkpoint(
            path,
            "cpu",
        )


def test_load_checkpoint_requires_settings_dictionary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.pt"

    torch.save(
        {
            "model_state_dict": {},
            "settings": [],
        },
        path,
    )

    with pytest.raises(TypeError):
        generative_experiment.load_checkpoint(
            path,
            "cpu",
        )


def test_load_checkpoint_returns_dictionary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"

    expected = {
        "model_state_dict": {
            "weight": torch.tensor([1.0]),
        },
        "settings": {
            "latent_dim": 2,
        },
    }

    torch.save(
        expected,
        path,
    )

    result = generative_experiment.load_checkpoint(
        path,
        "cpu",
    )

    assert result.keys() == expected.keys()
    assert result["settings"] == expected["settings"]
    assert torch.equal(
        result["model_state_dict"]["weight"],
        expected["model_state_dict"]["weight"],
    )


# ============================================================
# restore_vqvae_settings
# ============================================================


def test_restore_vqvae_settings_uses_stored_configuration() -> None:
    original = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=4,
        num_embeddings=16,
        use_ema=True,
        use_variance=True,
    )

    stored = asdict(original)

    restored = (
        generative_experiment.restore_vqvae_settings(
            stored,
            "cpu",
        )
    )

    assert restored.latent_dim == 4
    assert restored.num_embeddings == 16
    assert restored.use_ema is True
    assert restored.use_variance is True
    assert restored.device == "cpu"


def test_restore_vqvae_settings_overrides_device() -> None:
    stored = asdict(
        VQVAEExperimentSettings(
            device="cpu",
        )
    )

    restored = (
        generative_experiment.restore_vqvae_settings(
            stored,
            "cuda",
        )
    )

    assert restored.device == "cuda"


# ============================================================
# load_model
# ============================================================


def test_load_model_restores_state_dict(
    monkeypatch: pytest.MonkeyPatch,
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.loaded_state = None
            self.device = None
            self.eval_called = False

        def load_state_dict(self, state_dict):
            self.loaded_state = state_dict

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    model = FakeModel()

    expected_state = {
        "weight": torch.tensor(
            [1.0, 2.0]
        )
    }

    checkpoint = {
        "model_state_dict": expected_state,
        "settings": asdict(
            VQVAEExperimentSettings(
                device="cpu",
                latent_dim=2,
                num_embeddings=8,
                use_ema=True,
                use_variance=True,
            )
        ),
    }

    monkeypatch.setattr(
        generative_experiment,
        "create_model",
        lambda input_dim, settings: model,
    )

    result = generative_experiment.load_model(
        checkpoint,
        input_dim=2,
        device="cpu",
    )

    assert result is model
    assert result.loaded_state.keys() == (
        expected_state.keys()
    )
    assert torch.equal(
        result.loaded_state["weight"],
        expected_state["weight"],
    )
    assert result.device == "cpu"
    assert result.eval_called is True


# ============================================================
# create_training_loader
# ============================================================


def test_create_training_loader_uses_training_split(
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    expected_loader = object()

    def fake_prepare_dataset(settings):
        captured["settings"] = settings

        return (
            "mnist",
            expected_loader,
            None,
            None,
        )

    monkeypatch.setattr(
        generative_experiment,
        "prepare_dataset",
        fake_prepare_dataset,
    )

    result = (
        generative_experiment.create_training_loader(
            experiment_settings
        )
    )

    assert result is expected_loader

    settings = captured["settings"]

    assert settings.use_cifar10 is False
    assert settings.seed == experiment_settings.seed
    assert settings.device == experiment_settings.device
    assert settings.data_root == experiment_settings.data_root
    assert settings.train_batch_size == (
        experiment_settings.batch_size
    )


# ============================================================
# collect_training_codeword_indices
# ============================================================


def test_collect_training_codeword_indices_preserves_order() -> None:
    class FakeQuantizer:
        def assign(self, z_e):
            return z_e[:, 0].long()

    class FakeModel:
        quantizer = FakeQuantizer()

        def eval(self):
            return self

        def encode(self, batch):
            return batch

    loader = [
        torch.tensor(
            [
                [2.0, 0.0],
                [5.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [1.0, 0.0],
            ]
        ),
    ]

    result = (
        generative_experiment.collect_training_codeword_indices(
            FakeModel(),
            loader,
            "cpu",
        )
    )

    expected = torch.tensor(
        [2, 5, 1],
        dtype=torch.long,
    )

    assert torch.equal(
        result,
        expected,
    )


def test_collect_training_codeword_indices_rejects_non_tensor_batch() -> None:
    class FakeModel:
        def eval(self):
            return self

    with pytest.raises(TypeError):
        generative_experiment.collect_training_codeword_indices(
            FakeModel(),
            [[1, 2]],
            "cpu",
        )


def test_collect_training_codeword_indices_rejects_empty_loader() -> None:
    class FakeModel:
        def eval(self):
            return self

    with pytest.raises(RuntimeError):
        generative_experiment.collect_training_codeword_indices(
            FakeModel(),
            [],
            "cpu",
        )


# ============================================================
# estimate_codeword_prior
# ============================================================


def test_estimate_codeword_prior_normalizes_histogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQuantizer:
        def compute_statistics(self, indices):
            assert torch.equal(
                indices,
                torch.tensor(
                    [0, 1, 1, 2],
                    dtype=torch.long,
                ),
            )

            return {
                "entropy": 1.5,
                "rate": 1.5,
                "perplexity": 2.5,
                "utilization": 0.75,
                "active_codewords": 3,
                "dead_codewords": 1,
                "histogram": torch.tensor(
                    [1.0, 2.0, 1.0, 0.0]
                ),
            }

    class FakeModel:
        quantizer = FakeQuantizer()

    monkeypatch.setattr(
        generative_experiment,
        "collect_training_codeword_indices",
        lambda model, loader, device: torch.tensor(
            [0, 1, 1, 2],
            dtype=torch.long,
        ),
    )

    probabilities, statistics = (
        generative_experiment.estimate_codeword_prior(
            FakeModel(),
            loader=[],
            device="cpu",
        )
    )

    expected = torch.tensor(
        [0.25, 0.50, 0.25, 0.00]
    )

    assert torch.allclose(
        probabilities,
        expected,
    )

    assert statistics["entropy"] == 1.5
    assert statistics["rate"] == 1.5
    assert statistics["active_codewords"] == 3


def test_estimate_codeword_prior_rejects_zero_histogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQuantizer:
        def compute_statistics(self, indices):
            return {
                "histogram": torch.zeros(4),
            }

    class FakeModel:
        quantizer = FakeQuantizer()

    monkeypatch.setattr(
        generative_experiment,
        "collect_training_codeword_indices",
        lambda model, loader, device: torch.tensor(
            [0],
            dtype=torch.long,
        ),
    )

    with pytest.raises(RuntimeError):
        generative_experiment.estimate_codeword_prior(
            FakeModel(),
            loader=[],
            device="cpu",
        )


# ============================================================
# Latent sampling
# ============================================================


def test_sample_generated_latents_delegates_to_sampler(
    monkeypatch: pytest.MonkeyPatch,
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
) -> None:
    expected_latents = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    expected_indices = torch.tensor(
        [2, 7],
        dtype=torch.long,
    )

    captured = {}

    def fake_sampler(
        *,
        model,
        probabilities,
        n_samples,
        noise_scale,
    ):
        captured["model"] = model
        captured["probabilities"] = probabilities
        captured["n_samples"] = n_samples
        captured["noise_scale"] = noise_scale

        return (
            expected_latents,
            expected_indices,
        )

    monkeypatch.setattr(
        generative_experiment,
        "sample_from_codebook",
        fake_sampler,
    )

    fake_model = object()
    probabilities = torch.tensor(
        [0.25, 0.75]
    )

    result = (
        generative_experiment.sample_generated_latents(
            fake_model,
            probabilities,
            n_samples=2,
            noise_scale=0.5,
        )
    )

    assert result[0] is expected_latents
    assert result[1] is expected_indices
    assert captured["model"] is fake_model
    assert torch.equal(
        captured["probabilities"],
        probabilities,
    )
    assert captured["n_samples"] == 2
    assert captured["noise_scale"] == 0.5


# ============================================================
# generate_samples
# ============================================================


def test_generate_samples_decodes_sampled_latents(
    monkeypatch: pytest.MonkeyPatch,
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
) -> None:
    latent_samples = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    indices = torch.tensor(
        [4, 7],
        dtype=torch.long,
    )

    generated = torch.tensor(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
        ]
    )

    class FakeModel:
        def decode(self, latents):
            assert latents is latent_samples
            return generated

    def fake_sample_generated_latents(
        model,
        probabilities,
        n_samples,
        noise_scale,
    ):
        assert n_samples == experiment_settings.num_samples
        assert noise_scale == experiment_settings.noise_scale

        return (
            latent_samples,
            indices,
        )


    monkeypatch.setattr(
        generative_experiment,
        "sample_generated_latents",
        fake_sample_generated_latents,
    )

    result_generated, result_latents, result_indices = (
        generative_experiment.generate_samples(
            FakeModel(),
            torch.tensor([0.5, 0.5]),
            experiment_settings,
        )
    )

    assert torch.equal(
        result_generated,
        generated,
    )
    assert result_latents is latent_samples
    assert result_indices is indices


# ============================================================
# visualize_generations
# ============================================================


def test_visualize_generations_uses_visualizer(
    monkeypatch: pytest.MonkeyPatch,
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
) -> None:
    captured = {}

    generations_fig = Mock()

    class FakeVisualizer:
        def __init__(self, settings) -> None:
            captured["settings"] = settings

        def plot_generations(
            self,
            images,
            n_images,
            title,
        ):
            captured["images"] = images
            captured["n_images"] = n_images
            captured["title"] = title

            return generations_fig

    monkeypatch.setattr(
        generative_experiment,
        "Visualizer",
        FakeVisualizer,
    )

    generated = torch.zeros(
        4,
        784,
    )

    generative_experiment.visualize_generations(
        generated,
        experiment_settings,
    )

    assert captured["settings"] is not None

    assert torch.equal(
        torch.from_numpy(
            captured["images"]
        ),
        generated,
    )

    assert captured["n_images"] == (
        experiment_settings.num_samples
    )

    assert captured["title"] == (
        "Samples generated from "
        "the learned codeword prior"
    )

    generations_fig.show.assert_called_once_with()


# ============================================================
# run_generation_round
# ============================================================


def test_run_generation_round_connects_generation_and_visualization(
    monkeypatch: pytest.MonkeyPatch,
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
) -> None:
    captured = {}

    generated = torch.zeros(
        4,
        784,
    )

    latents = torch.zeros(
        4,
        2,
    )

    indices = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    monkeypatch.setattr(
        generative_experiment,
        "generate_samples",
        lambda model, probabilities, settings: (
            generated,
            latents,
            indices,
        ),
    )

    monkeypatch.setattr(
        generative_experiment,
        "visualize_generations",
        lambda generated_value, settings: captured.update(
            {
                "generated": generated_value,
                "settings": settings,
            }
        ),
    )

    generative_experiment.run_generation_round(
        model=object(),
        probabilities=torch.tensor(
            [0.25, 0.25, 0.25, 0.25]
        ),
        settings=experiment_settings,
    )

    assert captured["generated"] is generated
    assert captured["settings"] is experiment_settings


# ============================================================
# run_experiment orchestration
# ============================================================


def test_run_experiment_does_not_use_test_data(
    monkeypatch: pytest.MonkeyPatch,
    experiment_settings:
    generative_experiment.GenerativeExperimentSettings,
) -> None:
    captured = {
        "generation_rounds": 0,
    }

    checkpoint = {
        "model_state_dict": {},
        "settings": {
            "seed": 42,
            "device": "cpu",
        },
    }

    training_batch = torch.zeros(
        2,
        784,
    )

    fake_model = object()

    monkeypatch.setattr(
        generative_experiment,
        "load_checkpoint",
        lambda path, device: checkpoint,
    )

    monkeypatch.setattr(
        generative_experiment,
        "create_training_loader",
        lambda settings: [training_batch],
    )

    monkeypatch.setattr(
        generative_experiment,
        "load_model",
        lambda checkpoint, input_dim, device: fake_model,
    )

    monkeypatch.setattr(
        generative_experiment,
        "estimate_codeword_prior",
        lambda model, loader, device: (
            torch.tensor([1.0]),
            {
                "rate": 0.0,
                "entropy": 0.0,
                "perplexity": 1.0,
                "utilization": 1.0,
                "active_codewords": 1,
                "dead_codewords": 0,
                "histogram": torch.tensor([1.0]),
            },
        ),
    )

    def fake_generation_round(
        model,
        probabilities,
        settings,
    ):
        captured["generation_rounds"] += 1
        captured["model"] = model
        captured["probabilities"] = probabilities
        captured["settings"] = settings

    monkeypatch.setattr(
        generative_experiment,
        "run_generation_round",
        fake_generation_round,
    )

    generative_experiment.run_experiment(
        experiment_settings
    )

    assert captured["generation_rounds"] == 1
    assert captured["model"] is fake_model
    assert captured["settings"] is experiment_settings