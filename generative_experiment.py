"""
generative_experiment.py
========================

Generative experiment for a trained Statistical VQ-VAE.

The model is used without encoding any test image.

Generative process
------------------

First sample a discrete codeword index

    k ~ p(k)

where p(k) is estimated from the training-set codeword assignments.

Then sample a continuous latent representation

    z ~ N(mu_k, Sigma_k)

according to the statistical codebook.

Finally decode

    x_gen = D(z).

This distinguishes genuine generation from stochastic reconstruction.

Part of:

    Toward Statistical Quantization for
    Wake--Sleep Neural Networks
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import torch

from classes.visualizer import Visualizer, VisualizationSettings
from data_preprocessing.prepare import prepare_dataset
from model import create_model
from settings import VQVAEExperimentSettings
from classes.vqvae import VQVAE
from latent_sampling import sample_from_codebook
from reproducibility import set_seed

import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class GenerativeExperimentSettings:
    """
    Configuration for the generative experiment.

    Dataset, checkpoint, and device settings are kept separate from the
    model configuration stored in the trained checkpoint.
    """

    seed: int = 42
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_cifar10: bool = False
    data_root: str = "./data"

    checkpoint_directory: Path = Path("checkpoints")

    num_samples: int = 32
    noise_scale: float = 0.5
    batch_size: int = 128

    interactive: bool = True

    def __post_init__(self) -> None:
        """Validate experiment settings."""
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                "device must be either 'cpu' or 'cuda'."
            )

        if self.num_samples < 1:
            raise ValueError(
                "num_samples must be positive."
            )

        if self.noise_scale < 0.0:
            raise ValueError(
                "noise_scale must be non-negative."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )


# ============================================================
# Checkpoint handling
# ============================================================


def checkpoint_path(
    settings: GenerativeExperimentSettings,
) -> Path:
    """
    Return the dataset-specific final checkpoint path.
    """
    dataset_name = (
        "cifar10"
        if settings.use_cifar10
        else "mnist"
    )

    return (
        settings.checkpoint_directory
        / f"vqvae_{dataset_name}.pt"
    )


def load_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> dict:
    """
    Load and validate a VQ-VAE checkpoint.

    Returns
    -------
    dict
        Loaded checkpoint dictionary.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected a dictionary checkpoint."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    if "settings" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "serialized 'settings'."
        )

    if not isinstance(
        checkpoint["settings"],
        dict,
    ):
        raise TypeError(
            "Checkpoint 'settings' must be a dictionary."
        )

    return checkpoint


def restore_vqvae_settings(
    stored_settings: dict,
    device: str,
) -> VQVAEExperimentSettings:
    """
    Reconstruct VQ-VAE experiment settings from a checkpoint.

    The serialized checkpoint settings are used for model architecture and
    training configuration. The runtime device is supplied explicitly so
    that a checkpoint created on one device can be evaluated on another.
    """
    field_names = {
        field.name
        for field in fields(VQVAEExperimentSettings)
    }

    kwargs = {
        name: value
        for name, value in stored_settings.items()
        if name in field_names
    }

    if "checkpoint_directory" in kwargs:
        kwargs["checkpoint_directory"] = Path(
            kwargs["checkpoint_directory"]
        )

    if "results_path" in kwargs:
        kwargs["results_path"] = Path(
            kwargs["results_path"]
        )

    kwargs["device"] = device

    return VQVAEExperimentSettings(
        **kwargs
    )


def load_model(
    checkpoint: dict,
    input_dim: int,
    device: str,
) -> VQVAE:
    """
    Reconstruct and restore the trained VQ-VAE.

    The model configuration is reconstructed from the settings stored in
    the checkpoint.
    """
    settings = restore_vqvae_settings(
        checkpoint["settings"],
        device,
    )

    model = create_model(
        input_dim=input_dim,
        settings=settings,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model

def create_training_loader(
    settings: GenerativeExperimentSettings,
):
    """
    Create the training loader used to estimate the empirical codeword prior.

    Only training data are used by this experiment.
    """
    vqvae_settings = VQVAEExperimentSettings(
        seed=settings.seed,
        device=settings.device,
        use_cifar10=settings.use_cifar10,
        data_root=settings.data_root,
        train_batch_size=settings.batch_size,
        validation_batch_size=settings.batch_size,
        test_batch_size=settings.batch_size,
    )

    _, train_loader, _, _ = prepare_dataset(
        vqvae_settings
    )

    return train_loader


@torch.no_grad()
def collect_training_codeword_indices(
    model: VQVAE,
    loader: Iterable[torch.Tensor],
    device: str,
) -> torch.Tensor:
    """
    Collect codeword assignments for the complete training set.
    """
    model.eval()

    all_indices: list[torch.Tensor] = []

    for batch in loader:
        if not torch.is_tensor(batch):
            raise TypeError(
                "Expected the training loader to return tensors."
            )

        batch = batch.to(device)

        z_e = model.encode(batch)

        indices = model.quantizer.assign(z_e)

        all_indices.append(
            indices.detach().cpu()
        )

    if not all_indices:
        raise RuntimeError(
            "No training observations were processed."
        )

    return torch.cat(
        all_indices,
        dim=0,
    )


@torch.no_grad()
def estimate_codeword_prior(
    model: VQVAE,
    loader: Iterable[torch.Tensor],
    device: str,
) -> tuple[torch.Tensor, dict]:
    """
    Estimate the empirical discrete prior p(k) from training assignments.
    """
    indices = collect_training_codeword_indices(
        model,
        loader,
        device,
    )

    statistics = model.quantizer.compute_statistics(
        indices.to(device)
    )

    histogram = (
        statistics["histogram"]
        .detach()
        .cpu()
    )

    total = histogram.sum()

    if total <= 0:
        raise RuntimeError(
            "Cannot estimate a codeword prior from zero assignments."
        )

    probabilities = (
        histogram / total
    )

    return probabilities, statistics


@torch.no_grad()
def sample_generated_latents(
    model: VQVAE,
    probabilities: torch.Tensor,
    n_samples: int,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample latent vectors from the empirical codeword prior.
    """
    return sample_from_codebook(
        model=model,
        probabilities=probabilities,
        n_samples=n_samples,
        noise_scale=noise_scale,
    )


@torch.no_grad()
def generate_samples(
    model: VQVAE,
    probabilities: torch.Tensor,
    settings: GenerativeExperimentSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate and decode a batch of observations.
    """
    latents, indices = sample_generated_latents(
        model=model,
        probabilities=probabilities,
        n_samples=settings.num_samples,
        noise_scale=settings.noise_scale,
    )

    generated = model.decode(
        latents
    )

    return (
        generated.detach().cpu(),
        latents,
        indices,
    )


def visualize_generations(
    generated: torch.Tensor,
    settings: GenerativeExperimentSettings,
) -> None:
    """Visualize generated observations."""
    visualizer = Visualizer(
        VisualizationSettings()
    )

    generations_fig = visualizer.plot_generations(
        generated.numpy(),
        n_images=settings.num_samples,
        title=(
            "Samples generated from "
            "the learned codeword prior"
        ),
    )
    generations_fig.show()
    plt.show(block=True)


def print_prior_statistics(
    statistics: dict,
) -> None:
    """
    Print empirical codeword-prior statistics.
    """
    print()
    print("Empirical latent prior")
    print("-" * 60)

    print(
        f"{'Entropy / Rate':20s}: "
        f"{statistics['rate']:.6f}"
    )

    print(
        f"{'Perplexity':20s}: "
        f"{statistics['perplexity']:.6f}"
    )

    print(
        f"{'Utilization':20s}: "
        f"{statistics['utilization']:.6f}"
    )

    print(
        f"{'Active codewords':20s}: "
        f"{statistics['active_codewords']}"
    )

    print(
        f"{'Dead codewords':20s}: "
        f"{statistics['dead_codewords']}"
    )

    print()
    print("Training codeword histogram:")
    print(
        statistics["histogram"]
    )


def print_generation_statistics(
    latents: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """
    Print the sampled codeword indices and latent statistics.
    """
    print()
    print("Generated codeword indices:")
    print(
        indices.detach().cpu()
    )

    print()
    print("Generated latent statistics")
    print("-" * 60)

    print(
        "Mean:",
        latents.mean().item(),
    )

    print(
        "Std:",
        latents.std().item(),
    )

def run_generation_round(
    model: VQVAE,
    probabilities: torch.Tensor,
    settings: GenerativeExperimentSettings,
) -> None:
    """Perform and visualize one generation round."""
    print(
        "\nSampling latent representations..."
    )

    generated, latents, indices = generate_samples(
        model,
        probabilities,
        settings,
    )

    print(
        "\nDecoding generated samples..."
    )

    print_generation_statistics(
        latents,
        indices,
    )

    visualize_generations(
        generated,
        settings,
    )

def run_experiment(
    settings: GenerativeExperimentSettings,
) -> None:
    """
    Run the generative experiment.

    Only training data are used to estimate the empirical codeword prior.
    No test observation is encoded.
    """
    set_seed(
        settings.seed
    )

    checkpoint_file = checkpoint_path(
        settings
    )

    print()
    print("=" * 70)
    print("VQ-VAE Generative Experiment")
    print("=" * 70)

    print(
        "Device:",
        settings.device,
    )

    print(
        "Dataset:",
        "CIFAR-10"
        if settings.use_cifar10
        else "MNIST",
    )

    print(
        "Checkpoint:",
        checkpoint_file,
    )

    print(
        "Samples:",
        settings.num_samples,
    )

    print(
        "Noise scale:",
        settings.noise_scale,
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print(
        "Loading checkpoint..."
    )

    checkpoint = load_checkpoint(
        checkpoint_file,
        settings.device,
    )

    stored_settings = checkpoint[
        "settings"
    ]

    print()
    print(
        "Stored settings:"
    )

    for key, value in stored_settings.items():
        print(
            f"{key:20s}: {value}"
        )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    print()
    print(
        "Loading training data..."
    )

    training_loader = create_training_loader(
        settings
    )

    first_batch = next(
        iter(training_loader)
    )

    if not torch.is_tensor(first_batch):
        raise TypeError(
            "Expected the training loader to return tensors."
        )

    if first_batch.ndim != 2:
        raise ValueError(
            "Expected flattened training observations."
        )

    input_dim = first_batch.shape[-1]

    # --------------------------------------------------------
    # Restore model
    # --------------------------------------------------------

    print()
    print(
        "Reconstructing trained model..."
    )

    model = load_model(
        checkpoint,
        input_dim,
        settings.device,
    )

    print()
    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Estimate empirical prior
    # --------------------------------------------------------

    print()
    print(
        "Estimating codeword prior "
        "from training data..."
    )

    probabilities, statistics = (
        estimate_codeword_prior(
            model,
            training_loader,
            settings.device,
        )
    )

    print_prior_statistics(
        statistics
    )

    # --------------------------------------------------------
    # Generation loop
    # --------------------------------------------------------

    while True:
        run_generation_round(
            model,
            probabilities,
            settings,
        )

        if not settings.interactive:
            break

        print()

        while True:
            answer = input(
                "Would you like further generations? (y/n): "
            ).strip().lower()

            if answer in {"y", "n"}:
                break

            print(
                'Please provide a valid answer. ("y" or "n")'
            )

        if answer == "n":
            break

    print()
    print(
        "Experiment finished."
    )

def main() -> None:
    """Run the generative experiment with default settings."""
    settings = GenerativeExperimentSettings(
        use_cifar10=False,
        noise_scale=1.0,
    )

    run_experiment(
        settings
    )


if __name__ == "__main__":
    main()