"""
unsupervised_dataset.py
=======================

Dataset wrapper for unsupervised learning.

Many torchvision datasets return

    (sample, target)

while unsupervised learning algorithms only require

    sample.

This wrapper transparently removes the targets.

Example
-------

dataset = MNIST(...)

dataset = UnsupervisedDataset(dataset)

loader = DataLoader(dataset)

for batch in loader:

    # batch has shape (batch_size, ...)
    ...

Part of:

    Toward Statistical Quantization for
    Wake--Sleep Neural Networks
"""

from __future__ import annotations

from torch.utils.data import Dataset


# ============================================================
# Dataset Wrapper
# ============================================================

class UnsupervisedDataset(Dataset):

    """
    Converts a supervised dataset into an unsupervised one.

    If the wrapped dataset returns

        (sample, target)

    only the sample is returned.

    If the wrapped dataset already returns only
    a sample, it is passed through unchanged.
    """

    def __init__(
        self,
        dataset: Dataset,
    ):

        self.dataset = dataset


    # --------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------

    def __len__(
        self,
    ) -> int:

        return len(self.dataset) # type: ignore


    # --------------------------------------------------------
    # Access one sample
    # --------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ):

        sample = self.dataset[index]

        #
        # torchvision:
        #
        #     (x, y)
        #
        if isinstance(sample, tuple):

            return sample[0]

        #
        # already unsupervised
        #
        return sample