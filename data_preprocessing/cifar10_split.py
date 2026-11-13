"""
cifar10_split.py
================

Train/validation/test preprocessing for CIFAR-10.

The official CIFAR-10 training partition contains 50,000 images.
It is deterministically split into training and validation subsets.

The official 10,000-image CIFAR-10 test partition remains untouched.

The resulting loaders are compatible with the VQ-VAE implementation,
which expects each batch to contain only the input tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from data_preprocessing.dataset_split import (
    DatasetSplitSettings,
    split_dataset,
)


@dataclass(frozen=True)
class CIFAR10SplitSettings:
    """
    Configuration for the CIFAR-10 train/validation/test split.

    Parameters
    ----------
    root:
        Root directory containing the CIFAR-10 data.

    validation_fraction:
        Fraction of the official training partition assigned to
        the validation subset.

    seed:
        Seed used for the deterministic train/validation split.

    train_batch_size:
        Batch size used by the training loader.

    validation_batch_size:
        Batch size used by the validation loader.

    test_batch_size:
        Batch size used by the test loader.

    num_workers:
        Number of worker processes used by the data loaders.

    download:
        Whether torchvision should download CIFAR-10 when absent.
    """

    root: str = "./data"

    validation_fraction: float = 0.10

    seed: int = 42

    train_batch_size: int = 128

    validation_batch_size: int = 128

    test_batch_size: int = 128

    num_workers: int = 0

    download: bool = True


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset wrapper returning flattened tensors.

    Images are converted to tensors with values in ``[0, 1]`` and
    flattened from shape ``[3, 32, 32]`` to ``[3072]``.
    """

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
    ) -> None:
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=self.transform,
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        """
        Return one flattened image tensor.
        """
        image, _ = self.dataset[index]

        return image.reshape(-1)


class CIFAR10Split:
    """
    Create deterministic CIFAR-10 train/validation/test loaders.

    Only the official training partition is split.

    With the default validation fraction:

        50,000 official training images
            -> 45,000 training
            ->  5,000 validation

        10,000 official test images
            -> 10,000 test
    """

    def __init__(
        self,
        settings: CIFAR10SplitSettings,
    ) -> None:
        self.settings = settings

        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate loader and split configuration."""
        settings = self.settings

        if not 0.0 < settings.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must satisfy "
                "0 < validation_fraction < 1."
            )

        if settings.train_batch_size <= 0:
            raise ValueError(
                "train_batch_size must be positive."
            )

        if settings.validation_batch_size <= 0:
            raise ValueError(
                "validation_batch_size must be positive."
            )

        if settings.test_batch_size <= 0:
            raise ValueError(
                "test_batch_size must be positive."
            )

        if settings.num_workers < 0:
            raise ValueError(
                "num_workers must be non-negative."
            )

    def create(
        self,
    ) -> tuple[
        DataLoader,
        DataLoader,
        DataLoader,
    ]:
        """
        Create train, validation, and test data loaders.

        Returns
        -------
        tuple
            Training, validation, and test loaders.
        """
        settings = self.settings

        full_train_dataset = CIFAR10Dataset(
            root=settings.root,
            train=True,
            download=settings.download,
        )

        train_dataset, validation_dataset = split_dataset(
            full_train_dataset,
            DatasetSplitSettings(
                validation_fraction=settings.validation_fraction,
                seed=settings.seed,
            ),
        )

        test_dataset = CIFAR10Dataset(
            root=settings.root,
            train=False,
            download=settings.download,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.train_batch_size,
            shuffle=True,
            num_workers=settings.num_workers,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=settings.validation_batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=settings.test_batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
        )

        return (
            train_loader,
            validation_loader,
            test_loader,
        )


def create_cifar10_split(
    settings: CIFAR10SplitSettings | None = None,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create CIFAR-10 loaders using the supplied or default settings.
    """
    if settings is None:
        settings = CIFAR10SplitSettings()

    return CIFAR10Split(settings).create()