"""
stochastic_generation.py
========================

Experimental evaluation for the Results subsections

    * Latent Uncertainty and Stochastic Decoding
    * Conditional Generation

The script loads a trained Statistical VQ-VAE checkpoint and does not perform
any optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch

from classes.image_preparation import prepare_images
from classes.vqvae import VQVAE
from data_preprocessing.prepare import prepare_dataset
from latent_sampling import sample_codeword_latents
from model import create_model
from reproducibility import set_seed
from settings import VQVAEExperimentSettings


# ============================================================
# Experiment configuration
# ============================================================


@dataclass(frozen=True)
class StochasticGenerationSettings:
    """
    Configuration specific to the stochastic-generation experiments.

    Model, dataset, device, checkpoint, and reproducibility settings are
    inherited from :class:`VQVAEExperimentSettings`.
    """

    n_uncertainty_observations: int = 8
    n_conditional_observations: int = 8
    n_conditional_samples: int = 8
    n_columns: int = 8

    # alpha = 0: deterministic conditional mean
    # alpha > 0: stochastic sampling around the conditional mean
    alphas: tuple[float, ...] = (
        0.0,
        2.0,
        4.0,
    )

    conditional_alpha: float = 4.0

    def __post_init__(self) -> None:
        """Validate stochastic-generation settings."""
        if self.n_uncertainty_observations < 1:
            raise ValueError(
                "n_uncertainty_observations must be positive."
            )

        if self.n_conditional_observations < 1:
            raise ValueError(
                "n_conditional_observations must be positive."
            )

        if self.n_conditional_samples < 1:
            raise ValueError(
                "n_conditional_samples must be positive."
            )

        if self.n_columns < 1:
            raise ValueError(
                "n_columns must be positive."
            )

        if not self.alphas:
            raise ValueError(
                "alphas must not be empty."
            )

        if any(
            alpha < 0.0
            for alpha in self.alphas
        ):
            raise ValueError(
                "All alpha values must be non-negative."
            )

        if self.conditional_alpha < 0.0:
            raise ValueError(
                "conditional_alpha must be non-negative."
            )


# ============================================================
# Checkpoint handling
# ============================================================


def checkpoint_path(
    settings: VQVAEExperimentSettings,
    dataset_name: str,
) -> Path:
    """
    Return the final checkpoint path for a dataset.

    Parameters
    ----------
    settings:
        Main VQ-VAE experiment settings.

    dataset_name:
        Dataset name returned by the dataset preparation pipeline.

    Returns
    -------
    Path
        Path to the dataset-specific final checkpoint.
    """
    return (
        settings.checkpoint_directory
        /
        f"vqvae_{dataset_name}.pt"
    )


def load_model(
    checkpoint: Path,
    input_dim: int,
    settings: VQVAEExperimentSettings,
) -> VQVAE:
    """
    Restore a trained VQ-VAE model from a final checkpoint.

    No optimization state is restored because this experiment performs
    evaluation only.

    Parameters
    ----------
    checkpoint:
        Path to the saved checkpoint.

    input_dim:
        Flattened input dimension of the dataset.

    settings:
        Main VQ-VAE experiment settings.

    Returns
    -------
    VQVAE
        Restored model in evaluation mode.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    loaded = torch.load(
        checkpoint,
        map_location=settings.device,
        weights_only=False,
    )

    if not isinstance(
        loaded,
        dict,
    ):
        raise TypeError(
            "Expected a dictionary checkpoint."
        )

    if "model_state_dict" not in loaded:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model = create_model(
        input_dim=input_dim,
        settings=settings,
    )

    model.load_state_dict(
        loaded["model_state_dict"]
    )

    model.to(
        settings.device
    )

    model.eval()

    return model


# ============================================================
# Evaluation split selection
# ============================================================


def select_evaluation_loader(
    train_loader,
    validation_loader,
    test_loader,
    evaluation_split: str,
):
    """
    Select the requested evaluation data loader.

    Parameters
    ----------
    train_loader:
        Training data loader.

    validation_loader:
        Validation data loader.

    test_loader:
        Test data loader.

    evaluation_split:
        Requested evaluation split. Supported values are
        ``"validation"`` and ``"test"``.

    Returns
    -------
    DataLoader
        Loader corresponding to the requested evaluation split.

    Raises
    ------
    ValueError
        If an unsupported split name is supplied.
    """
    if evaluation_split == "validation":
        return validation_loader

    if evaluation_split == "test":
        return test_loader

    raise ValueError(
        "evaluation_split must be "
        "'validation' or 'test'."
    )


# ============================================================
# Evaluation-set selection
# ============================================================


@torch.no_grad()
def collect_evaluation_examples(
    evaluation_loader: Iterable[torch.Tensor],
    n_examples: int,
) -> torch.Tensor:
    """
    Collect the first ``n_examples`` observations.

    The original evaluation-set order is preserved.

    Parameters
    ----------
    evaluation_loader:
        Iterable yielding flattened evaluation observations.

    n_examples:
        Number of observations to collect.

    Returns
    -------
    torch.Tensor
        Collected observations on CPU.
    """
    if n_examples < 1:
        raise ValueError(
            "n_examples must be positive."
        )

    examples: list[torch.Tensor] = []

    total = 0

    for batch in evaluation_loader:

        if not torch.is_tensor(batch):
            raise TypeError(
                "Expected the evaluation loader to return tensors."
            )

        remaining = (
            n_examples
            -
            total
        )

        if remaining <= 0:
            break

        selected = batch[
            :remaining
        ]

        examples.append(
            selected.detach().cpu()
        )

        total += selected.shape[0]

        if total >= n_examples:
            break

    if not examples:
        raise RuntimeError(
            "No evaluation examples were collected."
        )

    return torch.cat(
        examples,
        dim=0,
    )


# ============================================================
# Conditional latent statistics
# ============================================================


@torch.no_grad()
def conditional_means(
    model: VQVAE,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Return the conditional mean latent for each codeword index.
    """
    return model.quantizer.embedding_mean(
        indices
    )


@torch.no_grad()
def conditional_stds(
    model: VQVAE,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Return conditional component-wise standard deviations.
    """
    if not model.quantizer.settings.use_variance:
        raise RuntimeError(
            "Variance sampling is disabled in the loaded model."
        )

    if not hasattr(
        model.quantizer,
        "embedding_logvar",
    ):
        raise AttributeError(
            "The loaded quantizer has no embedding_logvar."
        )

    logvar = model.quantizer.embedding_logvar(
        indices
    )

    variance = torch.clamp(
        torch.exp(logvar),
        min=0.0,
    )

    return torch.sqrt(
        variance + 1e-8
    )


# ============================================================
# Visualization: alpha comparison
# ============================================================


def plot_uncertainty_comparison(
    generated: dict[float, torch.Tensor],
    *,
    n_columns: int = 8,
    title: str,
) -> None:
    """
    Plot stochastic decoding for several uncertainty scales.

    Each column corresponds to the same source observation across rows.
    """
    if not generated:
        raise ValueError(
            "No generated samples were supplied."
        )

    if n_columns < 1:
        raise ValueError(
            "n_columns must be positive."
        )

    alphas = list(
        generated
    )

    n_images = min(
        generated[alpha].shape[0]
        for alpha in alphas
    )

    if n_images < 1:
        raise ValueError(
            "Generated tensors contain no observations."
        )

    n_columns = min(
        n_columns,
        n_images,
    )

    fig, axes = plt.subplots(
        len(alphas),
        n_columns,
        figsize=(
            1.8 * n_columns,
            2.0 * len(alphas),
        ),
        squeeze=False,
    )

    for row, alpha in enumerate(
        alphas
    ):
        images = prepare_images(
            generated[alpha]
        )

        for column in range(
            n_columns
        ):
            image = images[column]

            axes[
                row,
                column,
            ].imshow(
                image,
                cmap=(
                    "gray"
                    if image.ndim == 2
                    else None
                ),
            )

            axes[
                row,
                column,
            ].axis("off")

        axes[
            row,
            0,
        ].set_ylabel(
            rf"$\alpha={alpha:g}$",
            fontsize=10,
        )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    plt.show()


# ============================================================
# Visualization: fixed-codeword conditional generation
# ============================================================


def plot_conditional_generations(
    observations: torch.Tensor,
    generations: torch.Tensor,
    indices: torch.Tensor,
    *,
    title: str,
) -> None:
    """
    Plot observations and stochastic generations from fixed codewords.

    Each row corresponds to one observation. The first column contains the
    observed input, while the remaining columns contain samples generated
    from the same fixed codeword assignment.
    """
    if observations.ndim != 2:
        raise ValueError(
            "observations must have shape "
            "(n_observations, input_dim)."
        )

    if generations.ndim != 3:
        raise ValueError(
            "generations must have shape "
            "(n_samples, n_observations, input_dim)."
        )

    if indices.ndim != 1:
        raise ValueError(
            "indices must be one-dimensional."
        )

    n_observations = (
        observations.shape[0]
    )

    n_samples = (
        generations.shape[0]
    )

    if n_observations < 1:
        raise ValueError(
            "observations must not be empty."
        )

    if n_samples < 1:
        raise ValueError(
            "generations must contain at least one sample."
        )

    if indices.shape[0] != n_observations:
        raise ValueError(
            "Observation and index counts do not match."
        )

    if generations.shape[1] != n_observations:
        raise ValueError(
            "Observation and generation counts do not match."
        )

    original_images = prepare_images(
        observations
    )

    flattened_generations = generations.reshape(
        -1,
        generations.shape[-1],
    )

    generated_images = prepare_images(
        flattened_generations
    )

    fig, axes = plt.subplots(
        n_observations,
        n_samples + 1,
        figsize=(
            1.8 * (n_samples + 1),
            2.0 * n_observations,
        ),
        squeeze=False,
    )

    for observation_index in range(
        n_observations
    ):
        original = original_images[
            observation_index
        ]

        axes[
            observation_index,
            0,
        ].imshow(
            original,
            cmap=(
                "gray"
                if original.ndim == 2
                else None
            ),
        )

        axes[
            observation_index,
            0,
        ].axis("off")

        axes[
            observation_index,
            0,
        ].set_ylabel(
            f"$K={int(indices[observation_index].item())}$",
            fontsize=10,
        )

        for sample_index in range(
            n_samples
        ):
            flat_index = (
                sample_index * n_observations
                +
                observation_index
            )

            image = generated_images[
                flat_index
            ]

            axes[
                observation_index,
                sample_index + 1,
            ].imshow(
                image,
                cmap=(
                    "gray"
                    if image.ndim == 2
                    else None
                ),
            )

            axes[
                observation_index,
                sample_index + 1,
            ].axis("off")

    axes[
        0,
        0,
    ].set_title(
        "Original"
    )

    for sample_index in range(
        n_samples
    ):
        axes[
            0,
            sample_index + 1,
        ].set_title(
            f"Sample {sample_index + 1}"
        )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    plt.show()


# ============================================================
# Main experiment
# ============================================================


def run_experiment(
    vqvae_settings: VQVAEExperimentSettings,
    generation_settings: StochasticGenerationSettings,
    evaluation_split: str = "test",
) -> None:
    """
    Run the stochastic-generation experiments.

    The trained model is never optimized or modified.

    Parameters
    ----------
    vqvae_settings:
        Main VQ-VAE experiment settings.

    generation_settings:
        Stochastic-generation experiment settings.

    evaluation_split:
        Dataset split used for the stochastic-generation experiment.
        Supported values are ``"validation"`` and ``"test"``.
        The default is ``"test"`` to preserve the original behavior.
    """
    set_seed(
        vqvae_settings.seed
    )

    (
        dataset_name,
        train_loader,
        validation_loader,
        test_loader,
    ) = prepare_dataset(
        vqvae_settings
    )

    evaluation_loader = select_evaluation_loader(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        evaluation_split=evaluation_split,
    )

    checkpoint = checkpoint_path(
        vqvae_settings,
        dataset_name,
    )

    dataset_label = (
        "CIFAR-10"
        if vqvae_settings.use_cifar10
        else "MNIST"
    )

    split_label = (
        evaluation_split.capitalize()
    )

    print(
        "=" * 70
    )

    print(
        "Statistical VQ-VAE stochastic generation experiment"
    )

    print(
        "=" * 70
    )

    print(
        "Device:",
        vqvae_settings.device,
    )

    print(
        "Dataset:",
        dataset_label,
    )

    print(
        "Evaluation split:",
        split_label,
    )

    print(
        "Checkpoint:",
        checkpoint,
    )

    # --------------------------------------------------------
    # Evaluation data
    # --------------------------------------------------------

    first_batch = next(
        iter(evaluation_loader)
    )

    if not torch.is_tensor(
        first_batch
    ):
        raise TypeError(
            "Expected the evaluation loader to return tensors."
        )

    if first_batch.ndim != 2:
        raise ValueError(
            "Expected flattened evaluation observations."
        )

    input_dim = (
        first_batch.shape[-1]
    )

    n_required = max(
        generation_settings.n_uncertainty_observations,
        generation_settings.n_conditional_observations,
    )

    observations = collect_evaluation_examples(
        evaluation_loader,
        n_required,
    ).to(
        vqvae_settings.device
    )

    # --------------------------------------------------------
    # Trained statistical VQ-VAE
    # --------------------------------------------------------

    model = load_model(
        checkpoint,
        input_dim,
        vqvae_settings,
    )

    # Encode and assign ONCE. These assignments remain fixed in both
    # experiments, separating the discrete state from local latent noise.
    with torch.no_grad():

        z_e = model.encode(
            observations
        )

        indices = model.quantizer.assign(
            z_e
        )

    print(
        "Selected codeword assignments:"
    )

    print(
        indices.detach().cpu()
    )

    # ========================================================
    # 1. Latent Uncertainty and Stochastic Decoding
    # ========================================================

    n_uncertainty = min(
        generation_settings.n_uncertainty_observations,
        observations.shape[0],
    )

    uncertainty_indices = indices[
        :n_uncertainty
    ]

    generated_by_alpha: dict[
        float,
        torch.Tensor,
    ] = {}

    print(
        "\nLatent uncertainty experiment"
    )

    means = conditional_means(
        model,
        uncertainty_indices,
    )

    conditional_std = conditional_stds(
        model,
        uncertainty_indices,
    )

    print(
        "Conditional std: "
        f"min={conditional_std.min().item():.8f}, "
        f"mean={conditional_std.mean().item():.8f}, "
        f"max={conditional_std.max().item():.8f}"
    )

    for alpha in (
        generation_settings.alphas
    ):

        latent_samples = sample_codeword_latents(
            model=model,
            indices=uncertainty_indices,
            noise_scale=alpha,
            n_samples=1,
        )

        decoded = model.decode(
            latent_samples[0]
        )

        generated_by_alpha[
            alpha
        ] = decoded.cpu()

        noise = (
            latent_samples
            -
            means.unsqueeze(0)
        )

        print(
            f"alpha={alpha:g}: "
            f"noise std={noise.std().item():.8f}"
        )

    plot_uncertainty_comparison(
        generated_by_alpha,
        n_columns=generation_settings.n_columns,
        title=(
            f"{dataset_label} "
            f"{evaluation_split} "
            "stochastic decoding at different uncertainty scales"
        ),
    )

    # ========================================================
    # 2. Conditional Generation
    # ========================================================

    n_conditional = min(
        generation_settings.n_conditional_observations,
        observations.shape[0],
    )

    conditional_observations = observations[
        :n_conditional
    ]

    conditional_indices = indices[
        :n_conditional
    ]

    print(
        "\nConditional generation experiment"
    )

    print(
        "Fixed codeword assignments:"
    )

    print(
        conditional_indices.detach().cpu()
    )

    conditional_latents = sample_codeword_latents(
        model=model,
        indices=conditional_indices,
        noise_scale=(
            generation_settings.conditional_alpha
        ),
        n_samples=(
            generation_settings.n_conditional_samples
        ),
    )

    conditional_generations = model.decode(
        conditional_latents.reshape(
            -1,
            conditional_latents.shape[-1],
        )
    ).reshape(
        generation_settings.n_conditional_samples,
        n_conditional,
        -1,
    )

    plot_conditional_generations(
        conditional_observations.cpu(),
        conditional_generations.cpu(),
        conditional_indices.cpu(),
        title=(
            f"{dataset_label} "
            f"{evaluation_split} "
            "conditional generation with fixed codewords"
        ),
    )

    print(
        "\nExperiment finished."
    )


def main() -> None:
    """Run stochastic-generation experiments using the default settings."""
    vqvae_settings = VQVAEExperimentSettings(
        use_cifar10=False,
        use_ema=True,
        use_variance=True,
        num_embeddings=2048,
    )

    generation_settings = StochasticGenerationSettings(
        alphas=(
            0.0,
            1.0,
            2.0,
        ),
        conditional_alpha=4.0,
    )

    run_experiment(
        vqvae_settings=vqvae_settings,
        generation_settings=generation_settings,
        evaluation_split="validation",
    )


if __name__ == "__main__":
    main()
