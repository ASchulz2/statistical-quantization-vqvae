"""
Dataset preparation for VQ-VAE experiments.

This module selects the configured dataset and creates
train, validation, and test data loaders.

MNIST and CIFAR-10 both follow the same logical protocol:

    official training partition
        -> training subset
        -> validation subset

    official test partition
        -> untouched test set
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from data_preprocessing.cifar10_split import (
    CIFAR10Split,
    CIFAR10SplitSettings,
)
from data_preprocessing.mnist_split import (
    MNISTSplit,
    MNISTSplitSettings,
)
from settings import VQVAEExperimentSettings


DatasetLoaders = tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]


def prepare_dataset(
    settings: VQVAEExperimentSettings,
) -> tuple[
    str,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create train, validation, and test loaders.

    Parameters
    ----------
    settings:
        VQ-VAE experiment configuration.

    Returns
    -------
    tuple
        A tuple containing

            dataset_name
            train_loader
            validation_loader
            test_loader

        where ``dataset_name`` is either ``"mnist"`` or
        ``"cifar10"``.

    Notes
    -----
    The official test partition is left untouched for both
    datasets. The validation subset is created only from
    the official training partition.
    """
    if settings.use_cifar10:
        dataset_name = "cifar10"

        split_settings = CIFAR10SplitSettings(
            root=settings.data_root,
            validation_fraction=settings.validation_fraction,
            seed=settings.seed,
            train_batch_size=settings.train_batch_size,
            validation_batch_size=settings.validation_batch_size,
            test_batch_size=settings.test_batch_size,
            num_workers=settings.num_workers,
            download=True,
        )

        (
            train_loader,
            validation_loader,
            test_loader,
        ) = CIFAR10Split(
            split_settings
        ).create()

    else:
        dataset_name = "mnist"

        split_settings = MNISTSplitSettings(
            root=settings.data_root,
            validation_fraction=settings.validation_fraction,
            seed=settings.seed,
            train_batch_size=settings.train_batch_size,
            validation_batch_size=settings.validation_batch_size,
            test_batch_size=settings.test_batch_size,
            num_workers=settings.num_workers,
            flatten=True,
        )

        (
            train_loader,
            validation_loader,
            test_loader,
        ) = MNISTSplit(
            split_settings
        ).create()

    return (
        dataset_name,
        train_loader,
        validation_loader,
        test_loader,
    )