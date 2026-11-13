from settings import VQVAEExperimentSettings
from model import create_model

def test_create_model_uses_custom_experiment_configuration() -> None:
    settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=6,
        num_embeddings=32,
        commitment_beta=0.75,
        use_ema=True,
        use_variance=True,
        ema_decay=0.90,
        encoder_hidden_dims=(
            64,
            32,
            16,
        ),
        decoder_hidden_dims=(
            16,
            32,
            64,
        ),
    )

    model = create_model(
        input_dim=100,
        settings=settings,
    )

    assert model.settings.encoder.input_dim == 100

    assert (
        model.settings.encoder.hidden_dims
        == (64, 32, 16)
    )

    assert (
        model.settings.encoder.latent_dim
        == 6
    )

    assert (
        model.settings.decoder.hidden_dims
        == (16, 32, 64)
    )

    assert (
        model.settings.decoder.latent_dim
        == 6
    )

    assert (
        model.settings.decoder.output_dim
        == 100
    )

    assert (
        model.settings.quantizer.num_embeddings
        == 32
    )

    assert (
        model.settings.quantizer.embedding_dim
        == 6
    )

    assert (
        model.settings.quantizer.commitment_cost
        == 0.75
    )

    assert (
        model.settings.quantizer.use_ema
        is True
    )

    assert (
        model.settings.quantizer.use_variance
        is True
    )

    assert (
        model.settings.quantizer.ema_decay
        == 0.90
    )


def test_create_model_has_consistent_latent_dimension() -> None:
    settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=5,
        num_embeddings=20,
    )

    model = create_model(
        input_dim=12,
        settings=settings,
    )

    assert (
        model.encoder.settings.latent_dim
        == model.quantizer.settings.embedding_dim
    )

    assert (
        model.quantizer.settings.embedding_dim
        == model.decoder.settings.latent_dim
    )


def test_create_model_has_expected_input_output_dimension() -> None:
    settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=4,
    )

    model = create_model(
        input_dim=37,
        settings=settings,
    )

    assert (
        model.encoder.settings.input_dim
        == 37
    )

    assert (
        model.decoder.settings.output_dim
        == 37
    )