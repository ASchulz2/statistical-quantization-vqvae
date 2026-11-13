from __future__ import annotations

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

import data_preprocessing.mnist_split as mnist_module
from data_preprocessing.mnist_split import (
    MNISTSplit,
    MNISTSplitSettings,
)


class DummyMNISTDataset(Dataset):
    """In-memory replacement for the MNIST dataset."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.size = 100 if settings.train else 20

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        return torch.zeros(784)


def test_mnist_split_creates_expected_loaders(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mnist_module,
        "MNISTDataset",
        DummyMNISTDataset,
    )

    settings = MNISTSplitSettings(
        validation_fraction=0.20,
        seed=42,
        train_batch_size=10,
        validation_batch_size=5,
        test_batch_size=4,
        num_workers=0,
        flatten=True,
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = MNISTSplit(settings).create()

    assert len(train_loader.dataset) == 80
    assert len(validation_loader.dataset) == 20
    assert len(test_loader.dataset) == 20

    assert train_loader.batch_size == 10
    assert validation_loader.batch_size == 5
    assert test_loader.batch_size == 4

    assert isinstance(
        train_loader.sampler,
        RandomSampler,
    )

    assert isinstance(
        validation_loader.sampler,
        SequentialSampler,
    )

    assert isinstance(
        test_loader.sampler,
        SequentialSampler,
    )


def test_mnist_split_is_deterministic(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mnist_module,
        "MNISTDataset",
        DummyMNISTDataset,
    )

    settings = MNISTSplitSettings(
        validation_fraction=0.20,
        seed=42,
    )

    loaders_a = MNISTSplit(settings).create()
    loaders_b = MNISTSplit(settings).create()

    train_a = loaders_a[0].dataset
    validation_a = loaders_a[1].dataset

    train_b = loaders_b[0].dataset
    validation_b = loaders_b[1].dataset

    assert train_a.indices == train_b.indices
    assert (
        validation_a.indices
        == validation_b.indices
    )