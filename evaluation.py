"""
Evaluation utilities for VQ-VAE experiments.

Model selection is performed using the validation set. The held-out
test set is evaluated separately and only when explicitly requested.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from classes.vqvae import VQVAE
from settings import VQVAEExperimentSettings


def _extract_targets(
    dataset: Dataset,
) -> torch.Tensor | None:
    """
    Extract targets from a dataset when they are available.

    The helper supports datasets exposing ``targets`` directly as well as
    wrappers such as ``Subset`` and custom datasets containing an underlying
    dataset in ``dataset``.

    Returns
    -------
    torch.Tensor or None
        Target labels in dataset order, or ``None`` when no targets are
        available.
    """
    targets = getattr(
        dataset,
        "targets",
        None,
    )

    if targets is not None:
        return torch.as_tensor(
            targets
        )

    underlying = getattr(
        dataset,
        "dataset",
        None,
    )

    if underlying is None:
        return None

    underlying_targets = _extract_targets(
        underlying
    )

    if underlying_targets is None:
        return None

    indices = getattr(
        dataset,
        "indices",
        None,
    )

    if indices is not None:
        return underlying_targets[
            torch.as_tensor(
                indices,
                dtype=torch.long,
            )
        ]

    return underlying_targets


@torch.no_grad()
def evaluate_model(
    model: VQVAE,
    data_loader: DataLoader,
    settings: VQVAEExperimentSettings,
    split_name: str = "dataset",
) -> dict:
    """
    Evaluate a selected VQ-VAE on a dataset split.

    Parameters
    ----------
    model:
        Validation-selected VQ-VAE model.

    data_loader:
        Data loader for the dataset split to evaluate.

    settings:
        Experiment configuration containing the evaluation device.

    split_name:
        Human-readable name of the evaluated split. It is used in
        error messages.

    Returns
    -------
    dict
        Evaluation outputs and scalar metrics. The returned dictionary
        contains the original inputs, reconstructions, codebook
        assignments, encoder latents, codeword histogram,
        reconstruction/codebook statistics, and optionally target
        labels when they are available from the underlying dataset.

    Raises
    ------
    RuntimeError
        If the data loader produces no samples or no codeword
        assignments, or if available targets do not match the number
        of evaluated samples.
    """
    model.eval()

    originals: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []

    for batch in data_loader:
        batch = batch.to(
            settings.device
        )

        output = model(
            batch
        )

        originals.append(
            batch.detach().cpu()
        )

        reconstructions.append(
            output.reconstruction.detach().cpu()
        )

        indices.append(
            output.indices.detach().cpu()
        )

        #
        # Reuse the encoder representation already produced
        # by the model forward pass.
        #

        latents.append(
            output.z_e.detach().cpu()
        )

    if not originals:
        raise RuntimeError(
            f"{split_name.capitalize()} loader produced no samples."
        )

    originals_tensor = torch.cat(
        originals,
        dim=0,
    )

    reconstructions_tensor = torch.cat(
        reconstructions,
        dim=0,
    )

    indices_tensor = torch.cat(
        indices,
        dim=0,
    )

    latents_tensor = torch.cat(
        latents,
        dim=0,
    )

    if indices_tensor.numel() == 0:
        raise RuntimeError(
            "No codeword assignments were obtained "
            f"during {split_name} evaluation."
        )

    #
    # Optional targets.
    #

    dataset = getattr(
        data_loader,
        "dataset",
        None,
    )

    targets = (
        _extract_targets(dataset)
        if dataset is not None
        else None
    )

    if targets is not None:
        if targets.shape[0] != originals_tensor.shape[0]:
            raise RuntimeError(
                "Number of targets does not match "
                f"the number of {split_name} samples."
            )

        targets = targets.detach().cpu()

    #
    # Reconstruction statistics.
    #

    distortion = torch.mean(
        (
            originals_tensor
            -
            reconstructions_tensor
        ).pow(2)
    )

    rmse = torch.sqrt(
        distortion
    )

    psnr = (
        -10.0
        *
        torch.log10(
            distortion + 1e-10
        )
    )

    #
    # Codebook statistics.
    #
    # VectorQuantizerTorch owns the canonical assignment
    # statistics implementation.
    #

    vq_statistics = (
        model.quantizer.compute_statistics(
            indices_tensor
        )
    )

    result = {
        "originals": originals_tensor,
        "reconstructions": reconstructions_tensor,
        "indices": indices_tensor,
        "latents": latents_tensor,
        "histogram": vq_statistics[
            "histogram"
        ],
        "distortion": distortion.item(),
        "distortion_db": psnr.item(),
        "rmse": rmse.item(),
        "psnr": psnr.item(),
        "rate": float(
            vq_statistics["rate"]
        ),
        "entropy": float(
            vq_statistics["entropy"]
        ),
        "perplexity": float(
            vq_statistics["perplexity"]
        ),
        "utilization": float(
            vq_statistics["utilization"]
        ),
        "active_codewords": int(
            vq_statistics["active_codewords"]
        ),
        "dead_codewords": int(
            vq_statistics["dead_codewords"]
        ),
        "targets": targets,
    }

    return result


@torch.no_grad()
def evaluate_validation_set(
    model: VQVAE,
    validation_loader: DataLoader,
    settings: VQVAEExperimentSettings,
) -> dict:
    """
    Evaluate the selected VQ-VAE on the validation set.

    Parameters
    ----------
    model:
        Validation-selected VQ-VAE model.

    validation_loader:
        Validation data loader.

    settings:
        Experiment configuration containing the evaluation device.

    Returns
    -------
    dict
        Validation outputs and scalar metrics.
    """
    return evaluate_model(
        model=model,
        data_loader=validation_loader,
        settings=settings,
        split_name="validation",
    )


@torch.no_grad()
def evaluate_test_set(
    model: VQVAE,
    test_loader: DataLoader,
    settings: VQVAEExperimentSettings,
) -> dict:
    """
    Evaluate the selected VQ-VAE on the held-out test set.

    Parameters
    ----------
    model:
        Validation-selected VQ-VAE model.

    test_loader:
        Test data loader.

    settings:
        Experiment configuration containing the evaluation device.

    Returns
    -------
    dict
        Test outputs and scalar metrics.
    """
    return evaluate_model(
        model=model,
        data_loader=test_loader,
        settings=settings,
        split_name="test",
    )