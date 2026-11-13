from __future__ import annotations

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

import data_preprocessing.cifar10_split as cifar10_module
from data_preprocessing.cifar10_split import (
    CIFAR10Split,
    CIFAR10SplitSettings,
)


class DummyCIFAR10Dataset(Dataset):
    """In-memory replacement for CIFAR-10."""

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
    ) -> None:
        self.root = root
        self.train = train
        self.download = download

        self.size = 100 if train else 20

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        return torch.zeros(3072)


def test_cifar10_split_creates_expected_loaders(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cifar10_module,
        "CIFAR10Dataset",
        DummyCIFAR10Dataset,
    )

    settings = CIFAR10SplitSettings(
        validation_fraction=0.20,
        seed=42,
        train_batch_size=10,
        validation_batch_size=5,
        test_batch_size=4,
        num_workers=0,
        download=False,
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = CIFAR10Split(settings).create()

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


def test_cifar10_split_is_deterministic(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cifar10_module,
        "CIFAR10Dataset",
        DummyCIFAR10Dataset,
    )

    settings = CIFAR10SplitSettings(
        validation_fraction=0.20,
        seed=42,
        download=False,
    )

    loaders_a = CIFAR10Split(settings).create()
    loaders_b = CIFAR10Split(settings).create()

    train_a = loaders_a[0].dataset
    validation_a = loaders_a[1].dataset

    train_b = loaders_b[0].dataset
    validation_b = loaders_b[1].dataset

    assert train_a.indices == train_b.indices
    assert (
        validation_a.indices
        == validation_b.indices
    )