"""
dataset_split.py
================

Generic deterministic train/validation dataset splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class DatasetSplitSettings:
    """
    Configuration for a deterministic train/validation split.

    Parameters
    ----------
    validation_fraction:
        Fraction of samples assigned to validation.

    seed:
        Random seed used to generate the permutation.
    """

    validation_fraction: float = 0.10

    seed: int = 42


def split_dataset(
    dataset: Dataset,
    settings: DatasetSplitSettings,
) -> tuple[Subset, Subset]:
    """
    Deterministically split a dataset into train and validation subsets.

    Parameters
    ----------
    dataset:
        Dataset to split.

    settings:
        Split configuration.

    Returns
    -------
    tuple
        ``(train_dataset, validation_dataset)``.

    Raises
    ------
    ValueError
        If the validation fraction is invalid or produces an empty
        train or validation subset.
    """
    if not 0.0 < settings.validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    n_samples = len(dataset)

    if n_samples < 2:
        raise ValueError(
            "Dataset must contain at least two samples."
        )

    n_validation = int(
        n_samples * settings.validation_fraction
    )

    n_train = n_samples - n_validation

    if n_validation <= 0:
        raise ValueError(
            "validation fraction produces no validation samples."
        )

    if n_train <= 0:
        raise ValueError(
            "validation fraction leaves no training samples."
        )

    generator = torch.Generator()
    generator.manual_seed(settings.seed)

    permutation = torch.randperm(
        n_samples,
        generator=generator,
    ).tolist()

    train_indices = permutation[:n_train]
    validation_indices = permutation[n_train:]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, validation_indices),
    )