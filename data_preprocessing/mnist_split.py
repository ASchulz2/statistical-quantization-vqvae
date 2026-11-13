"""
mnist_split.py
==============

Train/validation/test preprocessing for MNIST.

The official MNIST training partition is deterministically split into
training and validation subsets. The official test partition remains
untouched.

The resulting data loaders return flattened input tensors suitable for
the VQ-VAE implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from classes.vqvae_dataset import (
    MNISTDataset,
    VQVAEDatasetSettings,
)

from data_preprocessing.dataset_split import (
    DatasetSplitSettings,
    split_dataset,
)


@dataclass(frozen=True)
class MNISTSplitSettings:
    """
    Configuration for the MNIST train/validation/test split.

    Parameters
    ----------
    root:
        Root directory containing the MNIST data.

    validation_fraction:
        Fraction of the official training partition assigned to
        the validation subset.

    seed:
        Seed used for the deterministic train/validation split.

    train_batch_size:
        Batch size used by the training data loader.

    validation_batch_size:
        Batch size used by the validation data loader.

    test_batch_size:
        Batch size used by the test data loader.

    num_workers:
        Number of worker processes used by the data loaders.

    flatten:
        Whether MNIST images are flattened before being returned.
    """

    root: str = "./data"

    validation_fraction: float = 0.10

    seed: int = 42

    train_batch_size: int = 128

    validation_batch_size: int = 128

    test_batch_size: int = 128

    num_workers: int = 0

    flatten: bool = True


class MNISTSplit:
    """
    Create deterministic MNIST train, validation, and test loaders.

    Only the official MNIST training partition is split.

    With the default validation fraction (0.1):

        60,000 official training images
            -> 54,000 training
            ->  6,000 validation

        10,000 official test images
            -> 10,000 test

    Parameters
    ----------
    settings:
        Configuration controlling the dataset split and loaders.
    """

    def __init__(
        self,
        settings: MNISTSplitSettings,
    ) -> None:
        self.settings = settings

        self._validate_settings()

    def _validate_settings(self) -> None:
        """
        Validate the MNIST split configuration.

        Raises
        ------
        ValueError
            If a configuration value is invalid.
        """
        settings = self.settings

        if not (
            0.0
            < settings.validation_fraction
            < 1.0
        ):
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
            Training, validation, and test data loaders.

        Notes
        -----
        Only the official MNIST training partition is split.
        The official test partition remains untouched.
        """
        settings = self.settings

        #
        # Official training partition.
        #

        full_train_dataset = MNISTDataset(
            VQVAEDatasetSettings(
                root=settings.root,
                train=True,
                flatten=settings.flatten,
                shuffle=False,
            )
        )

        #
        # Deterministic train/validation split.
        #

        train_dataset, validation_dataset = split_dataset(
            full_train_dataset,
            DatasetSplitSettings(
                validation_fraction=(
                    settings.validation_fraction
                ),
                seed=settings.seed,
            ),
        )

        #
        # Official test partition.
        #

        test_dataset = MNISTDataset(
            VQVAEDatasetSettings(
                root=settings.root,
                train=False,
                flatten=settings.flatten,
                shuffle=False,
            )
        )

        #
        # Data loaders.
        #

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


def create_mnist_split(
    settings: MNISTSplitSettings | None = None,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create MNIST loaders using supplied or default settings.

    Parameters
    ----------
    settings:
        MNIST split configuration. When ``None``, default settings
        are used.

    Returns
    -------
    tuple
        Training, validation, and test data loaders.
    """
    if settings is None:
        settings = MNISTSplitSettings()

    return MNISTSplit(settings).create()