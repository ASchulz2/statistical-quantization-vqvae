from __future__ import annotations

import pytest
from torch.utils.data import Dataset

from data_preprocessing.dataset_split import (
    DatasetSplitSettings,
    split_dataset,
)


class DummyDataset(Dataset):
    """Small in-memory dataset for split tests."""

    def __init__(self, size: int) -> None:
        self.values = list(range(size))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


def test_split_preserves_all_samples() -> None:
    dataset = DummyDataset(100)

    train_dataset, validation_dataset = split_dataset(
        dataset,
        DatasetSplitSettings(
            validation_fraction=0.20,
            seed=42,
        ),
    )

    train_indices = set(train_dataset.indices)
    validation_indices = set(validation_dataset.indices)

    assert len(train_dataset) == 80
    assert len(validation_dataset) == 20

    assert train_indices.isdisjoint(
        validation_indices
    )

    assert train_indices | validation_indices == set(
        range(100)
    )


def test_split_is_deterministic() -> None:
    dataset = DummyDataset(100)

    settings = DatasetSplitSettings(
        validation_fraction=0.20,
        seed=42,
    )

    train_a, validation_a = split_dataset(
        dataset,
        settings,
    )

    train_b, validation_b = split_dataset(
        dataset,
        settings,
    )

    assert train_a.indices == train_b.indices
    assert (
        validation_a.indices
        == validation_b.indices
    )


def test_different_seed_changes_split() -> None:
    dataset = DummyDataset(100)

    train_a, validation_a = split_dataset(
        dataset,
        DatasetSplitSettings(
            validation_fraction=0.20,
            seed=42,
        ),
    )

    train_b, validation_b = split_dataset(
        dataset,
        DatasetSplitSettings(
            validation_fraction=0.20,
            seed=43,
        ),
    )

    assert (
        train_a.indices != train_b.indices
        or
        validation_a.indices != validation_b.indices
    )


@pytest.mark.parametrize(
    "validation_fraction",
    [
        -0.1,
        0.0,
        1.0,
        1.1,
    ],
)
def test_invalid_validation_fraction_raises(
    validation_fraction: float,
) -> None:
    dataset = DummyDataset(100)

    with pytest.raises(ValueError):
        split_dataset(
            dataset,
            DatasetSplitSettings(
                validation_fraction=validation_fraction,
            ),
        )


def test_too_small_dataset_raises() -> None:
    dataset = DummyDataset(1)

    with pytest.raises(ValueError):
        split_dataset(
            dataset,
            DatasetSplitSettings(
                validation_fraction=0.5,
            ),
        )


def test_fraction_that_produces_empty_validation_raises() -> None:
    dataset = DummyDataset(3)

    with pytest.raises(ValueError):
        split_dataset(
            dataset,
            DatasetSplitSettings(
                validation_fraction=0.1,
            ),
        )