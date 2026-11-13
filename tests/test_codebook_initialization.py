"""
Tests for VQ-VAE codebook initialization.
"""

from __future__ import annotations

import torch

from codebook_initialization import initialize_codebook
from settings import VQVAEExperimentSettings


class DummyQuantizer:
    """Test double for the vector quantizer."""

    def __init__(self) -> None:
        self.samples: torch.Tensor | None = None

    def initialize_from_samples(
        self,
        samples: torch.Tensor,
    ) -> None:
        self.samples = samples


class DummyModel:
    """Test double for the VQ-VAE model."""

    def __init__(self) -> None:
        self.quantizer = DummyQuantizer()
        self.eval_called = False
        self.encoded_batches: list[torch.Tensor] = []

    def eval(self) -> None:
        self.eval_called = True

    def encode(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        self.encoded_batches.append(batch)

        return batch[:, :2]


def test_initialize_codebook_uses_configured_number_of_batches() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
        kmeans_init_batches=2,
    )

    train_loader = [
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
        torch.tensor(
            [
                [7.0, 8.0, 9.0],
            ]
        ),
        torch.tensor(
            [
                [10.0, 11.0, 12.0],
            ]
        ),
    ]

    initialize_codebook(
        model=model,
        train_loader=train_loader,
        settings=settings,
    )

    assert model.eval_called is True
    assert len(model.encoded_batches) == 2

    assert model.quantizer.samples is not None

    assert torch.equal(
        model.quantizer.samples,
        torch.tensor(
            [
                [1.0, 2.0],
                [4.0, 5.0],
                [7.0, 8.0],
            ]
        ),
    )


def test_initialize_codebook_uses_device() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
        kmeans_init_batches=1,
    )

    train_loader = [
        torch.ones(
            2,
            4,
        )
    ]

    initialize_codebook(
        model=model,
        train_loader=train_loader,
        settings=settings,
    )

    assert len(model.encoded_batches) == 1
    assert model.encoded_batches[0].device.type == "cpu"


def test_initialize_codebook_raises_for_empty_loader() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
        kmeans_init_batches=2,
    )

    try:
        initialize_codebook(
            model=model,
            train_loader=[],
            settings=settings,
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "No latent samples were collected "
            "for codebook initialization."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_initialize_codebook_passes_grad_free_samples() -> None:
    class CheckingQuantizer:
        def __init__(self) -> None:
            self.samples: torch.Tensor | None = None

        def initialize_from_samples(
            self,
            samples: torch.Tensor,
        ) -> None:
            self.samples = samples
            assert samples.requires_grad is False

    model = DummyModel()
    model.quantizer = CheckingQuantizer()

    settings = VQVAEExperimentSettings(
        device="cpu",
        kmeans_init_batches=1,
    )

    train_loader = [
        torch.randn(
            4,
            8,
            requires_grad=True,
        )
    ]

    initialize_codebook(
        model=model,
        train_loader=train_loader,
        settings=settings,
    )

    assert model.quantizer.samples is not None