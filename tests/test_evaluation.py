"""
Tests for VQ-VAE evaluation.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import Subset

from evaluation import (
    _extract_targets,
    evaluate_model,
    evaluate_test_set,
    evaluate_validation_set,
)
from settings import VQVAEExperimentSettings


class DummyQuantizer:
    """Minimal quantizer for evaluation tests."""

    def __init__(self) -> None:
        self.settings = SimpleNamespace(
            num_embeddings=4,
        )

    def compute_statistics(
        self,
        indices: torch.Tensor,
    ) -> dict:
        histogram = torch.bincount(
            indices.flatten(),
            minlength=4,
        )

        active = histogram > 0

        probabilities = (
            histogram.float()
            /
            histogram.sum()
        )

        mask = probabilities > 0

        entropy = -torch.sum(
            probabilities[mask]
            *
            torch.log2(
                probabilities[mask]
            )
        )

        return {
            "histogram": histogram.float(),
            "entropy": entropy.item(),
            "rate": entropy.item(),
            "perplexity": torch.pow(
                2.0,
                entropy,
            ).item(),
            "utilization": active.float().mean().item(),
            "active_codewords": int(
                active.sum()
            ),
            "dead_codewords": int(
                (~active).sum()
            ),
        }


class DummyModel:
    """Minimal model producing deterministic reconstructions."""

    def __init__(self) -> None:
        self.quantizer = DummyQuantizer()
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(
        self,
        batch: torch.Tensor,
    ):
        z_e = batch[:, :2]

        reconstruction = batch * 0.5

        indices = torch.tensor(
            [0, 1],
            dtype=torch.long,
        )[: len(batch)]

        return SimpleNamespace(
            reconstruction=reconstruction,
            z_e=z_e,
            indices=indices,
        )


def test_evaluate_model_returns_expected_metrics() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
    )

    data_loader = [
        torch.ones(
            2,
            4,
        ),
    ]

    metrics = evaluate_model(
        model=model,
        data_loader=data_loader,
        settings=settings,
        split_name="validation",
    )

    assert model.eval_called is True

    assert metrics["originals"].shape == (
        2,
        4,
    )

    assert metrics["reconstructions"].shape == (
        2,
        4,
    )

    assert metrics["indices"].shape == (
        2,
    )

    assert metrics["latents"].shape == (
        2,
        2,
    )

    assert metrics["distortion"] == 0.25
    assert metrics["rmse"] == 0.5
    assert metrics["active_codewords"] == 2
    assert metrics["dead_codewords"] == 2


def test_evaluate_model_rejects_empty_loader() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
    )

    try:
        evaluate_model(
            model=model,
            data_loader=[],
            settings=settings,
            split_name="validation",
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "Validation loader produced no samples."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_evaluate_validation_set_uses_validation_split() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
    )

    validation_loader = [
        torch.ones(
            2,
            4,
        ),
    ]

    metrics = evaluate_validation_set(
        model=model,
        validation_loader=validation_loader,
        settings=settings,
    )

    assert model.eval_called is True
    assert metrics["distortion"] == 0.25
    assert metrics["active_codewords"] == 2


def test_evaluate_validation_set_rejects_empty_loader() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
    )

    try:
        evaluate_validation_set(
            model=model,
            validation_loader=[],
            settings=settings,
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "Validation loader produced no samples."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_evaluate_test_set_uses_test_split() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
    )

    test_loader = [
        torch.ones(
            2,
            4,
        ),
    ]

    metrics = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        settings=settings,
    )

    assert model.eval_called is True
    assert metrics["distortion"] == 0.25
    assert metrics["active_codewords"] == 2


def test_evaluate_test_set_rejects_empty_loader() -> None:
    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
    )

    try:
        evaluate_test_set(
            model=model,
            test_loader=[],
            settings=settings,
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "Test loader produced no samples."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_extract_targets_from_dataset() -> None:
    class FakeDataset:
        targets = [0, 1, 2, 3]

    result = _extract_targets(
        FakeDataset()
    )

    assert torch.equal(
        result,
        torch.tensor([0, 1, 2, 3]),
    )


def test_extract_targets_from_subset() -> None:
    class FakeDataset:
        targets = [10, 11, 12, 13]

    subset = Subset(
        FakeDataset(),
        [3, 1],
    )

    result = _extract_targets(
        subset
    )

    assert torch.equal(
        result,
        torch.tensor([13, 11]),
    )


def test_extract_targets_returns_none_when_unavailable() -> None:
    class FakeDataset:
        pass

    result = _extract_targets(
        FakeDataset()
    )

    assert result is None


def test_evaluate_test_set_returns_targets() -> None:
    class TargetDataset:
        targets = torch.tensor(
            [3, 7]
        )

    class TargetLoader:
        dataset = TargetDataset()

        def __iter__(self):
            yield torch.ones(
                2,
                4,
            )

    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
    )

    metrics = evaluate_test_set(
        model=model,
        test_loader=TargetLoader(),
        settings=settings,
    )

    assert torch.equal(
        metrics["targets"],
        torch.tensor([3, 7]),
    )


def test_evaluate_validation_set_returns_targets() -> None:
    class TargetDataset:
        targets = torch.tensor(
            [4, 9]
        )

    class TargetLoader:
        dataset = TargetDataset()

        def __iter__(self):
            yield torch.ones(
                2,
                4,
            )

    model = DummyModel()

    settings = VQVAEExperimentSettings(
        device="cpu",
    )

    metrics = evaluate_validation_set(
        model=model,
        validation_loader=TargetLoader(),
        settings=settings,
    )

    assert torch.equal(
        metrics["targets"],
        torch.tensor([4, 9]),
    )