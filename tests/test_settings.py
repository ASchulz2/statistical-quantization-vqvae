from pathlib import Path

import pytest

from settings import VQVAEExperimentSettings


def test_default_settings() -> None:
    settings = VQVAEExperimentSettings()

    assert settings.seed == 42
    assert settings.device in {"cpu", "cuda"}
    assert settings.use_cifar10 is False

    assert settings.epochs == 5
    assert settings.train_batch_size == 128
    assert settings.validation_batch_size == 128
    assert settings.test_batch_size == 128
    assert settings.learning_rate == 1e-3

    assert settings.latent_dim == 2
    assert settings.num_embeddings == 1024

    assert settings.use_ema is True
    assert settings.ema_decay == 0.95
    assert settings.use_variance is False
    assert settings.commitment_beta == 1.0
    assert settings.plot_codebook is False

    assert settings.validation_fraction == 0.10
    assert settings.kmeans_init_batches == 40

    assert settings.n_visualization_images == 32
    assert settings.n_generative_images == 32
    assert settings.noise_scale == 0.0

    assert settings.checkpoint_directory == Path("checkpoints")
    assert settings.results_path == Path("results/experiments.csv")


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("seed", -1),
        ("epochs", 0),
        ("train_batch_size", 0),
        ("validation_batch_size", 0),
        ("test_batch_size", 0),
        ("learning_rate", 0.0),
        ("latent_dim", 0),
        ("num_embeddings", 0),
        ("ema_decay", 0.0),
        ("commitment_beta", -1.0),
        ("validation_fraction", 0.0),
        ("kmeans_init_batches", 0),
        ("n_visualization_images", -1),
        ("n_generative_images", -1),
        ("noise_scale", -1.0),
    ],
)
def test_invalid_settings_raise_value_error(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        VQVAEExperimentSettings(
            **{field_name: value}
        )


def test_device_validation() -> None:
    with pytest.raises(ValueError):
        VQVAEExperimentSettings(
            device="tpu"
        )


def test_custom_settings() -> None:
    settings = VQVAEExperimentSettings(
        seed=89,
        device="cpu",
        use_cifar10=True,
        epochs=10,
        latent_dim=16,
        num_embeddings=512,
        noise_scale=0.5,
    )

    assert settings.seed == 89
    assert settings.device == "cpu"
    assert settings.use_cifar10 is True
    assert settings.epochs == 10
    assert settings.latent_dim == 16
    assert settings.num_embeddings == 512
    assert settings.noise_scale == 0.5