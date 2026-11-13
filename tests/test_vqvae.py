"""
Tests for the complete VQ-VAE model.
"""

from __future__ import annotations

import pytest
import torch

from classes.decoder import DecoderSettings
from classes.encoder import EncoderSettings
from classes.vector_quantizer_torch import (
    VectorQuantizerTorchSettings,
)
from classes.vqvae import (
    VQVAE,
    VQVAESettings,
)


def create_test_settings() -> VQVAESettings:
    """
    Create a small CPU-only VQ-VAE configuration for testing.
    """
    return VQVAESettings(
        encoder=EncoderSettings(
            input_dim=16,
            hidden_dims=(
                8,
                4,
            ),
            latent_dim=2,
            device="cpu",
        ),
        decoder=DecoderSettings(
            latent_dim=2,
            hidden_dims=(
                4,
                8,
            ),
            output_dim=16,
            device="cpu",
        ),
        quantizer=VectorQuantizerTorchSettings(
            num_embeddings=8,
            embedding_dim=2,
            commitment_cost=0.5,
            device="cpu",
            use_ema=False,
            use_variance=False,
        ),
    )


def create_test_model() -> VQVAE:
    """
    Create a small CPU-only VQ-VAE for testing.
    """
    return VQVAE(
        create_test_settings()
    )


def test_model_stores_settings() -> None:
    settings = create_test_settings()

    model = VQVAE(
        settings
    )

    assert model.settings is settings


def test_model_constructs_all_components() -> None:
    model = create_test_model()

    assert model.encoder is not None
    assert model.quantizer is not None
    assert model.decoder is not None


def test_encode_returns_expected_shape() -> None:
    model = create_test_model()

    x = torch.rand(
        4,
        16,
    )

    z_e = model.encode(x)

    assert z_e.shape == (
        4,
        2,
    )

    assert z_e.dtype == torch.float32


def test_quantize_returns_expected_shapes() -> None:
    model = create_test_model()

    z_e = torch.randn(
        4,
        2,
    )

    output = model.quantize(
        z_e
    )

    assert output.z_q.shape == (
        4,
        2,
    )

    assert output.indices.shape == (
        4,
    )

    assert output.vq_loss.ndim == 0
    assert output.commitment_loss.ndim == 0
    assert output.variance_loss.ndim == 0
    assert output.batch_entropy.ndim == 0
    assert output.batch_rate.ndim == 0
    assert output.batch_perplexity.ndim == 0
    assert output.batch_utilization.ndim == 0


def test_decode_returns_expected_shape() -> None:
    model = create_test_model()

    z_q = torch.randn(
        4,
        2,
    )

    reconstruction = model.decode(
        z_q
    )

    assert reconstruction.shape == (
        4,
        16,
    )

    assert reconstruction.dtype == torch.float32


def test_forward_returns_complete_output() -> None:
    model = create_test_model()

    x = torch.rand(
        4,
        16,
    )

    output = model(x)

    assert output.reconstruction.shape == (
        4,
        16,
    )

    assert output.z_e.shape == (
        4,
        2,
    )

    assert output.z_q.shape == (
        4,
        2,
    )

    assert output.indices.shape == (
        4,
    )

    assert output.vq_loss.ndim == 0
    assert output.commitment_loss.ndim == 0
    assert output.variance_loss.ndim == 0
    assert output.perplexity.ndim == 0
    assert output.utilization.ndim == 0


def test_forward_matches_explicit_pipeline() -> None:
    """
    Verify that forward() is equivalent to the public
    encode -> quantize -> decode pipeline.
    """
    torch.manual_seed(42)

    model = create_test_model()

    x = torch.rand(
        4,
        16,
    )

    z_e = model.encode(
        x
    )

    quantized = model.quantize(
        z_e
    )

    reconstruction = model.decode(
        quantized.z_q
    )

    torch.manual_seed(42)

    model = create_test_model()

    output = model(
        x
    )

    assert torch.allclose(
        output.z_e,
        z_e,
    )

    assert torch.equal(
        output.indices,
        quantized.indices,
    )

    assert torch.allclose(
        output.z_q,
        quantized.z_q,
    )

    assert torch.allclose(
        output.reconstruction,
        reconstruction,
    )


def test_forward_supports_single_sample() -> None:
    model = create_test_model()

    x = torch.rand(
        16,
    )

    output = model(x)

    assert output.reconstruction.shape == (
        16,
    )

    assert output.z_e.shape == (
        2,
    )

    assert output.z_q.shape == (
        2,
    )

    assert output.indices.ndim == 0


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        8,
    ],
)
def test_forward_supports_different_batch_sizes(
    batch_size: int,
) -> None:
    model = create_test_model()

    x = torch.rand(
        batch_size,
        16,
    )

    output = model(x)

    assert output.reconstruction.shape == (
        batch_size,
        16,
    )

    assert output.z_e.shape == (
        batch_size,
        2,
    )

    assert output.z_q.shape == (
        batch_size,
        2,
    )

    assert output.indices.shape == (
        batch_size,
    )