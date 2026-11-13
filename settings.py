from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


def _default_device() -> str:
    """Return the automatically selected PyTorch device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VQVAEExperimentSettings:
    """
    Configuration for a VQ-VAE experiment.

    Parameters
    ----------
    seed:
        Random seed used for experiment reproducibility.

    device:
        PyTorch device used for computation. If omitted, CUDA is selected
        when available and CPU otherwise.

    use_cifar10:
        Select CIFAR-10 instead of MNIST as the dataset.

    epochs:
        Number of training epochs.

    train_batch_size:
        Batch size used for training.

    validation_batch_size:
        Batch size used for validation.

    test_batch_size:
        Batch size used for testing.

    learning_rate:
        Optimizer learning rate.

    latent_dim:
        Dimension of the continuous latent representation.

    num_embeddings:
        Number of codewords in the vector-quantization codebook.

    use_ema:
        Whether exponential-moving-average codebook updates are enabled.

    ema_decay:
        Exponential-moving-average decay factor.

    use_variance:
        Whether variance-based latent modelling is enabled.

    commitment_beta:
        Commitment-loss coefficient of the VQ objective.

    plot_codebook:
        Whether the learned codebook should be visualized.

    validation_fraction:
        Fraction of the official training set reserved for validation.

    kmeans_init_batches:
        Number of batches used for K-Means codebook initialization.

    n_visualization_images:
        Number of images used for visualization.

    n_generative_images:
        Number of images generated for generative evaluation.

    noise_scale:
        Scale of stochastic latent noise. A value of zero produces
        deterministic codeword means.

    checkpoint_directory:
        Directory in which model checkpoints are stored.

    results_path:
        CSV file containing machine-readable experiment results.

    Raises
    ------
    ValueError
        If a configuration parameter is outside its valid range.
    """

    # Reproducibility
    seed: int = 42

    # Device
    device: str = field(default_factory=_default_device)

    # Training
    epochs: int = 5
    train_batch_size: int = 128
    validation_batch_size: int = 128
    test_batch_size: int = 128
    learning_rate: float = 1e-3

    # Latent representation
    latent_dim: int = 2

    # Model architecture
    encoder_hidden_dims: tuple[int, ...] = (
        512,
        256,
        256,
    )

    decoder_hidden_dims: tuple[int, ...] = (
        256,
        256,
        512,
    )

    # Codebook
    num_embeddings: int = 1024

    # Vector quantization
    use_ema: bool = True
    ema_decay: float = 0.95
    use_variance: bool = False
    commitment_beta: float = 1.0
    plot_codebook: bool = False

    # Validation
    validation_fraction: float = 0.10

    # K-Means initialization
    kmeans_init_batches: int = 40

    # Visualization
    n_visualization_images: int = 32
    n_generative_images: int = 32

    plot_validation_latent_space: bool = False

    # Stochastic latent generation
    noise_scale: float = 0.0

    # Dataset
    use_cifar10: bool = False
    data_root: str = "./data"
    num_workers: int = 0

    # ============================================================
    # Trainer
    # ============================================================

    print_every: int = 1

    save_best_checkpoint: bool = True

    best_checkpoint_name: str = "vqvae_best.pt"

    restore_metric: str = "val_distortion"

    restore_best_model: bool = True
    

    # Checkpoints
    checkpoint_directory: Path = field(
        default_factory=lambda: Path("checkpoints")
    )

    # Experimental results
    results_path: Path = field(
        default_factory=lambda: Path("results/experiments.csv")
    )

    def __post_init__(self) -> None:
        """Validate the experiment configuration."""
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                "device must be either 'cpu' or 'cuda'"
            )

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.train_batch_size <= 0:
            raise ValueError(
                "train_batch_size must be positive"
            )

        if self.validation_batch_size <= 0:
            raise ValueError(
                "validation_batch_size must be positive"
            )

        if self.test_batch_size <= 0:
            raise ValueError(
                "test_batch_size must be positive"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive"
            )

        if self.num_embeddings <= 0:
            raise ValueError(
                "num_embeddings must be positive"
            )

        if not 0.0 < self.ema_decay <= 1.0:
            raise ValueError(
                "ema_decay must satisfy 0 < ema_decay <= 1"
            )

        if self.commitment_beta < 0.0:
            raise ValueError(
                "commitment_beta must be non-negative"
            )

        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must satisfy 0 < fraction < 1"
            )

        if self.kmeans_init_batches <= 0:
            raise ValueError(
                "kmeans_init_batches must be positive"
            )

        if self.n_visualization_images < 0:
            raise ValueError(
                "n_visualization_images must be non-negative"
            )

        if self.n_generative_images < 0:
            raise ValueError(
                "n_generative_images must be non-negative"
            )

        if self.noise_scale < 0.0:
            raise ValueError(
                "noise_scale must be non-negative"
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers must be non-negative"
            )

        if not self.encoder_hidden_dims:
            raise ValueError(
                "encoder_hidden_dims must not be empty"
            )

        if any(
            dimension <= 0
            for dimension in self.encoder_hidden_dims
        ):
            raise ValueError(
                "encoder_hidden_dims must contain only "
                "positive dimensions"
            )

        if not self.decoder_hidden_dims:
            raise ValueError(
                "decoder_hidden_dims must not be empty"
            )

        if any(
            dimension <= 0
            for dimension in self.decoder_hidden_dims
        ):
            raise ValueError(
                "decoder_hidden_dims must contain only "
                "positive dimensions"
            )

        if self.print_every <= 0:
            raise ValueError(
                "print_every must be positive"
            )

        if not self.best_checkpoint_name.strip():
            raise ValueError(
                "best_checkpoint_name must not be empty"
            )

        if self.restore_metric not in {
            "val_loss",
            "val_distortion",
        }:
            raise ValueError(
                "restore_metric must be one of: "
                "'val_loss', 'val_distortion'"
            )