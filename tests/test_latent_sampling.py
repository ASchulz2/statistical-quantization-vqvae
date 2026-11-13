"""
Tests for statistical latent sampling.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from latent_sampling import (
    sample_quantized_latents,
)

from latent_sampling import sample_from_codebook


class DummyQuantizer:
    """Small quantizer test double."""

    def __init__(
        self,
        use_variance: bool,
    ) -> None:
        self.settings = SimpleNamespace(
            use_variance=use_variance
        )

        self.embedding_mean = nn.Embedding(
            2,
            2,
        )

        self.embedding_logvar = nn.Embedding(
            2,
            2,
        )

        with torch.no_grad():
            self.embedding_mean.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ]
                )
            )

            self.embedding_logvar.weight.fill_(
                0.0
            )

    def assign(
        self,
        z_e: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(
            z_e.shape[0],
            dtype=torch.long,
            device=z_e.device,
        )


def create_dummy_model(
    use_variance: bool,
) -> SimpleNamespace:
    """Create a minimal model compatible with the sampler."""
    return SimpleNamespace(
        quantizer=DummyQuantizer(
            use_variance=use_variance
        )
    )


def test_zero_noise_is_deterministic() -> None:
    model = create_dummy_model(
        use_variance=True
    )

    z_e = torch.randn(
        4,
        2,
    )

    result = sample_quantized_latents(
        model=model,
        z_e=z_e,
        noise_scale=0.0,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ]
    )

    assert torch.equal(
        result,
        expected,
    )


def test_variance_disabled_is_deterministic() -> None:
    model = create_dummy_model(
        use_variance=False
    )

    z_e = torch.randn(
        4,
        2,
    )

    result = sample_quantized_latents(
        model=model,
        z_e=z_e,
        noise_scale=1.0,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ]
    )

    assert torch.equal(
        result,
        expected,
    )


def test_stochastic_sampling_uses_variance() -> None:
    model = create_dummy_model(
        use_variance=True
    )

    with torch.no_grad():
        model.quantizer.embedding_logvar.weight.fill_(
            torch.log(
                torch.tensor(4.0)
            )
        )

    z_e = torch.randn(
        16,
        2,
    )

    torch.manual_seed(42)

    result = sample_quantized_latents(
        model=model,
        z_e=z_e,
        noise_scale=1.0,
    )

    expected_mean = torch.tensor(
        [1.0, 2.0]
    )

    assert result.shape == (
        16,
        2,
    )

    assert not torch.allclose(
        result,
        expected_mean.expand_as(result),
    )


def test_negative_noise_scale_raises() -> None:
    model = create_dummy_model(
        use_variance=True
    )

    with pytest.raises(ValueError):
        sample_quantized_latents(
            model=model,
            z_e=torch.randn(2, 2),
            noise_scale=-1.0,
        )


def test_missing_variance_embedding_raises() -> None:
    model = create_dummy_model(
        use_variance=True
    )

    del model.quantizer.embedding_logvar

    with pytest.raises(AttributeError):
        sample_quantized_latents(
            model=model,
            z_e=torch.randn(2, 2),
            noise_scale=1.0,
        )


def test_sample_from_codebook_rejects_stochastic_sampling_without_variance() -> None:
    class FakeSettings:
        num_embeddings = 2
        use_variance = False

    class FakeQuantizer:
        settings = FakeSettings()

        def __init__(self) -> None:
            self.embedding_mean = torch.nn.Embedding(
                2,
                2,
            )

    class FakeModel:
        def __init__(self) -> None:
            self.quantizer = FakeQuantizer()

    model = FakeModel()

    probabilities = torch.tensor(
        [0.5, 0.5],
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="Stochastic generation requires",
    ):
        sample_from_codebook(
            model=model,
            probabilities=probabilities,
            n_samples=4,
            noise_scale=0.5,
        )