from __future__ import annotations

from dataclasses import dataclass

import torch

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from classes.unsupervised_dataset import UnsupervisedDataset

from torchvision import datasets, transforms



# ============================================================
# Settings
# ============================================================

@dataclass
class VQVAEDatasetSettings:

    batch_size: int = 128

    shuffle: bool = True

    num_workers: int = 0

    train: bool = True

    flatten: bool = True

    root: str = "./data"



# ============================================================
# Dataset
# ============================================================

class MNISTDataset(Dataset):

    """
    MNIST wrapper for VQ-VAE.

    Returns:

        x

    where

        x.shape = [784]

    """

    def __init__(
        self,
        settings: VQVAEDatasetSettings,
    ):

        self.settings = settings


        self.transform = transforms.Compose(

            [

                transforms.ToTensor(),

            ]

        )


        self.dataset = datasets.MNIST(

            root=settings.root,

            train=settings.train,

            download=True,

            transform=self.transform,

        )



    def __len__(
        self,
    ):

        return len(self.dataset)



    def __getitem__(
        self,
        index: int,
    ):

        image, _ = self.dataset[index]


        if self.settings.flatten:

            image = image.reshape(-1)


        return image



# ============================================================
# DataLoader
# ============================================================

class VQVAEDataLoader:

    """
    Creates train/test loaders for VQ-VAE.
    """

    def __init__(
        self,
        settings: VQVAEDatasetSettings,
    ):

        self.settings = settings



    def create(
        self,
    ):

        dataset = MNISTDataset(

            self.settings

        )

        dataset = UnsupervisedDataset(dataset)

        loader = DataLoader(

            dataset,

            batch_size=self.settings.batch_size,

            shuffle=self.settings.shuffle,

            num_workers=self.settings.num_workers,

        )


        return loader