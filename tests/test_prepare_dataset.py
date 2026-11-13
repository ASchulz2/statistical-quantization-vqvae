from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from data_preprocessing.prepare import prepare_dataset
import data_preprocessing.prepare as prepare_module

from settings import VQVAEExperimentSettings



@dataclass
class FakeSplit:
    settings: object

    def create(self):
        return (
            "train",
            "validation",
            "test",
        )


def test_prepare_dataset_uses_mnist(
    monkeypatch,
) -> None:

    captured = {}

    def fake_mnist_split(settings):
        captured["settings"] = settings
        return FakeSplit(settings)

    monkeypatch.setattr(
        prepare_module,
        "MNISTSplit",
        fake_mnist_split,
    )

    settings = VQVAEExperimentSettings(
        use_cifar10=False,
        seed=89,
        data_root="./test-data",
        train_batch_size=64,
        validation_batch_size=32,
        test_batch_size=16,
        num_workers=2,
    )

    (
        dataset_name,
        train_loader,
        validation_loader,
        test_loader,
    ) = prepare_dataset(settings)

    assert dataset_name == "mnist"

    assert train_loader == "train"
    assert validation_loader == "validation"
    assert test_loader == "test"

    split_settings = captured["settings"]

    assert split_settings.root == "./test-data"
    assert split_settings.validation_fraction == 0.10
    assert split_settings.seed == 89
    assert split_settings.train_batch_size == 64
    assert split_settings.validation_batch_size == 32
    assert split_settings.test_batch_size == 16
    assert split_settings.num_workers == 2
    assert split_settings.flatten is True


def test_prepare_dataset_uses_cifar10(
    monkeypatch,
) -> None:

    captured = {}

    def fake_cifar10_split(settings):
        captured["settings"] = settings
        return FakeSplit(settings)

    monkeypatch.setattr(
        prepare_module,
        "CIFAR10Split",
        fake_cifar10_split,
    )

    settings = VQVAEExperimentSettings(
        use_cifar10=True,
        seed=89,
        data_root="./test-data",
        train_batch_size=64,
        validation_batch_size=32,
        test_batch_size=16,
        num_workers=2,
    )

    (
        dataset_name,
        train_loader,
        validation_loader,
        test_loader,
    ) = prepare_dataset(settings)

    assert dataset_name == "cifar10"

    assert train_loader == "train"
    assert validation_loader == "validation"
    assert test_loader == "test"

    split_settings = captured["settings"]

    assert split_settings.root == "./test-data"
    assert split_settings.validation_fraction == 0.10
    assert split_settings.seed == 89
    assert split_settings.train_batch_size == 64
    assert split_settings.validation_batch_size == 32
    assert split_settings.test_batch_size == 16
    assert split_settings.num_workers == 2
    assert split_settings.download is True