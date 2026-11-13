"""
Visualization utilities for VQ-VAE and related experiments.

The :class:`Visualizer` class provides plotting routines for:

    * latent spaces
    * sequences
    * autocorrelations
    * rate-distortion curves
    * quantization cells
    * reconstructions
    * reconstruction residuals
    * generated samples
    * decoded codebooks
    * training/validation metrics
    * recursive VQ-VAE refinements

Image layout conversion is delegated to
:mod:`classes.image_preparation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from classes.image_preparation import (
    flattened_to_image,
    prepare_images,
)
from matplotlib.figure import Figure


@dataclass(frozen=True)
class VisualizationSettings:
    """
    Configuration for visualization output.

    Parameters
    ----------
    figsize:
        Default Matplotlib figure size.

    dpi:
        Default Matplotlib figure resolution.
    """

    figsize: tuple[int, int] = (8, 6)

    dpi: int = 100

    def __post_init__(self) -> None:
        """Validate visualization configuration."""
        if len(self.figsize) != 2:
            raise ValueError(
                "figsize must contain exactly two values."
            )

        if any(
            value <= 0
            for value in self.figsize
        ):
            raise ValueError(
                "figsize values must be positive."
            )

        if self.dpi <= 0:
            raise ValueError(
                "dpi must be positive."
            )


class Visualizer:
    """
    Plot diagnostic information for VQ-VAE experiments.

    Parameters
    ----------
    settings:
        Visualization configuration.

    Notes
    -----
    Plotting methods display figures using Matplotlib. Unit tests
    should replace ``plt.show`` when figures must not be displayed.
    """

    def __init__(
        self,
        settings: VisualizationSettings,
    ) -> None:
        self.settings = settings

    def plot_latent_space(
        self,
        samples: np.ndarray,
        codewords: np.ndarray | None = None,
        targets: np.ndarray | None = None,
        title: str = "Latent space",
    ) -> Figure:
        """
        Create a two-dimensional latent-space figure.

        The figure is returned to the caller and is not displayed here.

        Parameters
        ----------
        samples:
            Latent samples with shape ``(N, 2)``.
        codewords:
            Optional codeword coordinates with shape ``(M, 2)``.
        targets:
            Optional target labels with shape ``(N,)``.
        title:
            Plot title.

        Raises
        ------
        ValueError
            If the latent representation is not two-dimensional.
            If targets do not contain one label per sample.
            If codewords are not two-dimensional.
        """
        if samples.ndim != 2:
            raise ValueError(
                "Expected samples with shape (N, D)."
            )

        if samples.shape[1] != 2:
            raise ValueError(
                "Latent space plot requires 2 dimensions."
            )

        if targets is not None:
            if targets.ndim != 1:
                raise ValueError(
                    "Targets must have shape (N,)."
                )

            if targets.shape[0] != samples.shape[0]:
                raise ValueError(
                    "Number of targets must match number of samples."
                )

        if codewords is not None:
            if (
                codewords.ndim != 2
                or
                codewords.shape[1] != 2
            ):
                raise ValueError(
                    "Codewords must have shape (N, 2)."
                )

        fig = plt.figure(
            figsize=self.settings.figsize,
            dpi=self.settings.dpi,
        )

        if targets is None:
            plt.scatter(
                samples[:, 0],
                samples[:, 1],
                s=10,
                alpha=1.0,
                label="Samples",
            )

        else:
            unique_targets = np.unique(
                targets
            )

            for target in unique_targets:
                mask = targets == target

                plt.scatter(
                    samples[mask, 0],
                    samples[mask, 1],
                    s=10,
                    alpha=1.0,
                    label=str(target),
                )

        if codewords is not None:
            plt.scatter(
                codewords[:, 0],
                codewords[:, 1],
                marker="x",
                s=100,
                label="Codewords",
            )

        plt.title(title)
        plt.xlabel("Dimension 0")
        plt.ylabel("Dimension 1")

        handles, labels = (
            plt.gca().get_legend_handles_labels()
        )

        if handles:
            plt.legend()

        plt.grid(True)

        return fig

    def plot_sequence(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        dimension: int = 0,
        title: str = "Quantization reconstruction",
    ) -> None:
        """
        Plot one sequence dimension before and after reconstruction.
        """
        if dimension < 0:
            raise ValueError(
                "dimension must be non-negative."
            )

        if original.ndim != 2:
            raise ValueError(
                "original must have shape (N, D)."
            )

        if reconstructed.ndim != 2:
            raise ValueError(
                "reconstructed must have shape (N, D)."
            )

        if dimension >= original.shape[1]:
            raise ValueError(
                "dimension is outside the original data."
            )

        if dimension >= reconstructed.shape[1]:
            raise ValueError(
                "dimension is outside the reconstructed data."
            )

        plt.figure(
            figsize=self.settings.figsize,
            dpi=self.settings.dpi,
        )

        plt.plot(
            original[:, dimension],
            label="Original",
        )

        plt.plot(
            reconstructed[:, dimension],
            label="Reconstructed",
        )

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        handles, labels = plt.gca().get_legend_handles_labels()

        if handles:
            plt.legend()
        plt.grid(True)
        plt.show()

    def plot_autocorrelation(
        self,
        values: np.ndarray,
        title: str = "Sampled autocorrelation",
    ) -> None:
        """
        Plot an autocorrelation sequence.
        """
        plt.figure(
            figsize=self.settings.figsize,
            dpi=self.settings.dpi,
        )

        plt.stem(
            np.arange(
                len(values)
            ),
            values,
        )

        plt.title(title)
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.show()

    def plot_rate_distortion(
        self,
        rates: list[float],
        distortions: list[float],
        title: str = "Rate-Distortion Curve",
    ) -> None:
        """
        Plot a rate-distortion curve.
        """
        if len(rates) != len(distortions):
            raise ValueError(
                "rates and distortions must have equal length."
            )

        plt.figure(
            figsize=self.settings.figsize,
            dpi=self.settings.dpi,
        )

        plt.plot(
            rates,
            distortions,
            marker="o",
        )

        plt.title(title)
        plt.xlabel("Rate [bits/sample]")
        plt.ylabel("Distortion")
        plt.grid(True)
        plt.show()

    def plot_rate_distortion_comparison(
        self,
        direct: tuple[list[float], list[float]],
        predictive: tuple[list[float], list[float]],
    ) -> None:
        """
        Plot direct and predictive rate-distortion curves.
        """
        if len(direct[0]) != len(direct[1]):
            raise ValueError(
                "Direct rates and distortions must have equal length."
            )

        if len(predictive[0]) != len(predictive[1]):
            raise ValueError(
                "Predictive rates and distortions must have equal length."
            )

        plt.figure(
            figsize=self.settings.figsize,
            dpi=self.settings.dpi,
        )

        plt.plot(
            direct[0],
            direct[1],
            marker="o",
            label="Direct VQ",
        )

        plt.plot(
            predictive[0],
            predictive[1],
            marker="x",
            label="Predictive VQ",
        )

        plt.xlabel("Rate [bits/sample]")
        plt.ylabel("Distortion")
        plt.title("Rate-Distortion Comparison")
        handles, labels = plt.gca().get_legend_handles_labels()

        if handles:
            plt.legend()
        plt.grid(True)
        plt.show()

    def plot_quantization_cells(
        self,
        samples: np.ndarray,
        indices: np.ndarray,
        codewords: np.ndarray | None = None,
        title: str = "Voronoi quantization",
    ) -> None:
        """
        Plot two-dimensional quantization assignments.
        """
        if samples.ndim != 2:
            raise ValueError(
                "Expected samples with shape (N, D)."
            )

        if samples.shape[1] != 2:
            raise ValueError(
                "Visualization requires 2D samples."
            )

        if len(indices) != len(samples):
            raise ValueError(
                "indices must contain one value per sample."
            )

        plt.figure(
            figsize=self.settings.figsize,
            dpi=self.settings.dpi,
        )

        scatter = plt.scatter(
            samples[:, 0],
            samples[:, 1],
            c=indices,
            s=15,
            alpha=0.5,
        )

        if codewords is not None:
            if (
                codewords.ndim != 2
                or
                codewords.shape[1] != 2
            ):
                raise ValueError(
                    "Codewords must have shape (N, 2)."
                )

            plt.scatter(
                codewords[:, 0],
                codewords[:, 1],
                marker="x",
                s=120,
                label="Codewords",
            )

        plt.colorbar(
            scatter,
            label="Codeword index",
        )

        plt.title(title)
        plt.xlabel("Dimension 0")
        plt.ylabel("Dimension 1")
        handles, labels = plt.gca().get_legend_handles_labels()

        if handles:
            plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def extract_rate_distortion(
        results: list[dict],
    ) -> tuple[list[float], list[float]]:
        """
        Extract rate-distortion values from result dictionaries.
        """
        rates = [
            float(item["rate_bits"])
            for item in results
        ]

        distortions = [
            float(item["distortion"])
            for item in results
        ]

        return (
            rates,
            distortions,
        )


    def plot_images(
        self,
        images: np.ndarray | torch.Tensor,
        n_images: int = 16,
        n_columns: int = 8,
        title: str = "Images",
    ) -> Figure:
        """
        Plot image examples in a grid and return the figure.

        Image layout conversion is delegated to ``prepare_images``.
        """
        self._validate_image_plot_arguments(
            n_images=n_images,
            n_columns=n_columns,
        )

        prepared_images = prepare_images(
            images[:n_images]
        )

        n = min(
            n_images,
            len(prepared_images),
        )

        if n == 0:
            raise ValueError(
                "No images available for visualization."
            )

        n_columns = min(
            n_columns,
            n,
        )

        n_rows = int(
            np.ceil(
                n
                /
                n_columns
            )
        )

        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(
                1.8 * n_columns,
                1.9 * n_rows,
            ),
            squeeze=False,
            dpi=self.settings.dpi,
        )

        for index in range(
            n_rows * n_columns
        ):
            row = index // n_columns
            column = index % n_columns

            ax = axes[
                row,
                column,
            ]

            if index >= n:
                ax.axis("off")
                continue

            image = prepared_images[index]

            ax.imshow(
                image,
                cmap=(
                    "gray"
                    if image.ndim == 2
                    else None
                ),
            )

            ax.axis("off")

        fig.suptitle(title)
        fig.tight_layout()

        return fig


    def plot_reconstructions(
        self,
        original: np.ndarray | torch.Tensor,
        reconstructed: np.ndarray | torch.Tensor,
        n_images: int = 16,
        n_columns: int = 8,
        title: str = "Reconstructions",
    ) -> Figure:
        """
        Create a figure showing original and reconstructed images
        in repeated row pairs.

        The figure is returned to the caller and is not displayed here.
        """
        self._validate_image_plot_arguments(
            n_images=n_images,
            n_columns=n_columns,
        )

        original_images = prepare_images(
            original[:n_images]
        )

        reconstructed_images = prepare_images(
            reconstructed[:n_images]
        )

        n = min(
            n_images,
            len(original_images),
            len(reconstructed_images),
        )

        if n == 0:
            raise ValueError(
                "No images available for visualization."
            )

        n_columns = min(
            n_columns,
            n,
        )

        n_pairs = int(
            np.ceil(
                n / n_columns
            )
        )

        n_rows = 2 * n_pairs

        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(
                1.8 * n_columns,
                2.0 * n_rows,
            ),
            squeeze=False,
            dpi=self.settings.dpi,
        )

        for pair in range(n_pairs):

            start = pair * n_columns
            end = min(
                start + n_columns,
                n,
            )

            original_row = 2 * pair
            reconstruction_row = original_row + 1

            for column in range(n_columns):

                index = start + column

                if index >= end:
                    axes[
                        original_row,
                        column,
                    ].axis("off")

                    axes[
                        reconstruction_row,
                        column,
                    ].axis("off")

                    continue

                original_image = original_images[
                    index
                ]

                reconstruction_image = (
                    reconstructed_images[index]
                )

                axes[
                    original_row,
                    column,
                ].imshow(
                    original_image,
                    cmap=(
                        "gray"
                        if original_image.ndim == 2
                        else None
                    ),
                )

                axes[
                    original_row,
                    column,
                ].axis("off")

                axes[
                    reconstruction_row,
                    column,
                ].imshow(
                    reconstruction_image,
                    cmap=(
                        "gray"
                        if reconstruction_image.ndim == 2
                        else None
                    ),
                )

                axes[
                    reconstruction_row,
                    column,
                ].axis("off")

            axes[
                original_row,
                0,
            ].set_ylabel(
                "Original",
                fontsize=10,
            )

            axes[
                reconstruction_row,
                0,
            ].set_ylabel(
                "Reconstruction",
                fontsize=10,
            )

        fig.suptitle(title)
        fig.tight_layout()

        return fig

    def plot_codebook_images(
        self,
        decoder,
        codewords: torch.Tensor,
        device: str = "cpu",
        n_columns: int = 16,
        title: str = "Decoded codebook",
    ) -> Figure:
        """
        Create a figure showing the decoded codebook images.

        The figure is returned to the caller and is not displayed here.

        Parameters
        ----------
        decoder:
            Trained VQ-VAE decoder.

        codewords:
            Codebook tensor with shape ``(M, latent_dim)``.

        device:
            Device used for decoding.

        n_columns:
            Number of images shown per row.

        title:
            Figure title.
        """
        if codewords.ndim != 2:
            raise ValueError(
                "Expected codewords with shape "
                "(M, latent_dim). "
                f"Got {tuple(codewords.shape)}."
            )

        if n_columns <= 0:
            raise ValueError(
                "n_columns must be positive."
            )

        codewords = codewords.to(
            device
        )

        was_training = decoder.training
        decoder.eval()

        try:
            with torch.no_grad():
                decoded = decoder(
                    codewords
                )
        finally:
            decoder.train(
                was_training
            )

        images = prepare_images(
            decoded
        )

        n_images = len(images)

        if n_images == 0:
            raise ValueError(
                "The codebook contains no codewords."
            )

        n_rows = int(
            np.ceil(
                n_images
                /
                n_columns
            )
        )

        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(
                1.8 * n_columns,
                1.8 * n_rows,
            ),
            squeeze=False,
            dpi=self.settings.dpi,
        )

        for row in range(n_rows):
            for column in range(n_columns):
                axes[
                    row,
                    column,
                ].axis("off")

        for index in range(n_images):

            row = index // n_columns
            column = index % n_columns

            image = images[index]

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
            ].set_title(
                f"{index}",
                fontsize=8,
            )

        fig.suptitle(title)
        fig.tight_layout()

        return fig

    def plot_metrics(
        self,
        history: dict[str, list[float]],
        metrics: list[str] | None = None,
        title_prefix: str = "Training",
    ) -> list[Figure]:
        """
        Create figures for scalar metrics stored in a history dictionary.

        Returns
        -------
        list[Figure]
            One figure for each valid metric that is plotted.
        """
        if metrics is None:
            metrics = list(
                history.keys()
            )

        figures: list[Figure] = []

        for metric in metrics:

            if metric not in history:
                continue

            values = history[
                metric
            ]

            if not isinstance(
                values,
                list,
            ):
                continue

            if len(values) == 0:
                continue

            fig = plt.figure(
                figsize=self.settings.figsize,
                dpi=self.settings.dpi,
            )

            plt.plot(
                np.arange(
                    len(values)
                ),
                values,
                marker="o",
            )

            plt.title(
                f"{title_prefix}: {metric}"
            )

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.grid(True)
            plt.tight_layout()

            figures.append(
                fig
            )

        return figures

    def plot_reconstruction_residuals(
        self,
        original: np.ndarray | torch.Tensor,
        reconstructed: np.ndarray | torch.Tensor,
        n_images: int = 16,
        n_columns: int = 8,
        title: str = "Reconstruction residuals",
    ) -> Figure:
        """
        Create a figure showing original images and reconstruction
        residuals.

        The figure is returned to the caller and is not displayed here.
        """
        self._validate_image_plot_arguments(
            n_images=n_images,
            n_columns=n_columns,
        )

        original_images = prepare_images(
            original[:n_images]
        )

        reconstructed_images = prepare_images(
            reconstructed[:n_images]
        )

        n = min(
            n_images,
            len(original_images),
            len(reconstructed_images),
        )

        if n == 0:
            raise ValueError(
                "No images available for visualization."
            )

        residual_images = (
            original_images[:n]
            -
            reconstructed_images[:n]
        )

        n_columns = min(
            n_columns,
            n,
        )

        n_pairs = int(
            np.ceil(
                n / n_columns
            )
        )

        n_rows = 2 * n_pairs

        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(
                1.8 * n_columns,
                2.0 * n_rows,
            ),
            squeeze=False,
            dpi=self.settings.dpi,
        )

        max_abs_residual = np.max(
            np.abs(
                residual_images
            )
        )

        if max_abs_residual == 0:
            max_abs_residual = 1.0

        for pair in range(n_pairs):

            start = pair * n_columns
            end = min(
                start + n_columns,
                n,
            )

            original_row = 2 * pair
            residual_row = original_row + 1

            for column in range(n_columns):

                index = start + column

                if index >= end:

                    axes[
                        original_row,
                        column,
                    ].axis("off")

                    axes[
                        residual_row,
                        column,
                    ].axis("off")

                    continue

                axes[
                    original_row,
                    column,
                ].imshow(
                    original_images[index],
                    cmap=(
                        "gray"
                        if original_images[index].ndim == 2
                        else None
                    ),
                )

                axes[
                    original_row,
                    column,
                ].axis("off")

                axes[
                    residual_row,
                    column,
                ].imshow(
                    residual_images[index],
                    cmap="coolwarm",
                    vmin=-max_abs_residual,
                    vmax=max_abs_residual,
                )

                axes[
                    residual_row,
                    column,
                ].axis("off")

            axes[
                original_row,
                0,
            ].set_ylabel(
                "Original",
                fontsize=10,
            )

            axes[
                residual_row,
                0,
            ].set_ylabel(
                "Residual",
                fontsize=10,
            )

        fig.suptitle(title)
        fig.tight_layout()

        return fig

    def plot_generations(
        self,
        images: np.ndarray | torch.Tensor,
        n_images: int = 16,
        n_columns: int = 8,
        title: str = "Generated samples",
    ) -> Figure:
        """
        Create a figure showing generated images.

        The figure is returned to the caller and is not displayed here.
        """
        self._validate_image_plot_arguments(
            n_images=n_images,
            n_columns=n_columns,
        )

        generated_images = prepare_images(
            images[:n_images]
        )

        n = min(
            n_images,
            len(generated_images),
        )

        if n == 0:
            raise ValueError(
                "No images available for visualization."
            )

        n_columns = min(
            n_columns,
            n,
        )

        n_rows = int(
            np.ceil(
                n / n_columns
            )
        )

        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(
                1.8 * n_columns,
                1.9 * n_rows,
            ),
            squeeze=False,
            dpi=self.settings.dpi,
        )

        for index in range(
            n_rows * n_columns
        ):

            row = index // n_columns
            column = index % n_columns

            if index >= n:
                axes[
                    row,
                    column,
                ].axis("off")

                continue

            image = generated_images[
                index
            ]

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

        fig.suptitle(title)
        fig.tight_layout()

        return fig

    def plot_training_validation_metrics(
        self,
        history: dict[str, list[float]],
        metrics: tuple[str, ...] = (
            "loss",
            "reconstruction_loss",
            "distortion",
            "perplexity",
            "utilization",
        ),
    ) -> list[Figure]:
        """
        Create training and validation metric figures.

        A validation series is expected under ``val_<metric>``.
        Metrics without both series are skipped.

        Returns
        -------
        list[Figure]
            One figure for each plotted metric.
            An empty list is returned when no metrics can be plotted.
        """
        figures: list[Figure] = []

        for metric in metrics:

            validation_metric = (
                f"val_{metric}"
            )

            if metric not in history:
                continue

            if validation_metric not in history:
                continue

            train_values = history[
                metric
            ]

            validation_values = history[
                validation_metric
            ]

            if len(train_values) != len(
                validation_values
            ):
                raise ValueError(
                    f"Training and validation series for "
                    f"'{metric}' must have equal length."
                )

            epochs = np.arange(
                len(train_values)
            )

            fig = plt.figure(
                figsize=self.settings.figsize,
                dpi=self.settings.dpi,
            )

            plt.plot(
                epochs,
                train_values,
                marker="o",
                label="Training",
            )

            plt.plot(
                epochs,
                validation_values,
                marker="x",
                label="Validation",
            )

            plt.xlabel("Epoch")

            plt.ylabel(
                metric.replace(
                    "_",
                    " ",
                ).title()
            )

            plt.title(
                "Training and validation: "
                +
                metric.replace(
                    "_",
                    " ",
                ).title()
            )

            handles, labels = (
                plt.gca().get_legend_handles_labels()
            )

            if handles:
                plt.legend()

            plt.grid(True)
            plt.tight_layout()

            figures.append(fig)

        return figures

    def plot_recursive_residuals(
        self,
        original: np.ndarray,
        reconstructions: list[np.ndarray],
        n_images: int = 8,
        title: str = "Recursive VQ-VAE residuals",
    ) -> None:
        """
        Plot residuals after each recursive refinement stage.

        Supports flattened MNIST, flattened CIFAR-10, and
        generic one-dimensional observations.
        """
        if len(reconstructions) == 0:
            raise ValueError(
                "At least one reconstruction stage is required."
            )

        if len(original) == 0:
            raise ValueError(
                "Original array must not be empty."
            )

        if n_images <= 0:
            raise ValueError(
                "n_images must be positive."
            )

        n = min(
            n_images,
            len(original),
        )

        for stage, reconstruction in enumerate(
            reconstructions
        ):

            if len(reconstruction) < n:
                raise ValueError(
                    f"Reconstruction stage {stage} "
                    f"contains only {len(reconstruction)} "
                    f"samples, but {n} are required."
                )

        n_stages = len(
            reconstructions
        )

        fig, axes = plt.subplots(
            n,
            n_stages,
            figsize=(
                3 * n_stages,
                3 * n,
            ),
            squeeze=False,
        )

        for row in range(n):

            for stage in range(
                n_stages
            ):

                residual = np.asarray(
                    original[row]
                    -
                    reconstructions[stage][row]
                )

                ax = axes[
                    row,
                    stage,
                ]

                if residual.size == 28 * 28:

                    image = residual.reshape(
                        28,
                        28,
                    )

                    vmax = np.max(
                        np.abs(image)
                    )

                    if vmax == 0.0:
                        vmax = 1.0

                    ax.imshow(
                        image,
                        cmap="seismic",
                        vmin=-vmax,
                        vmax=vmax,
                    )

                elif residual.size == (
                    32 * 32 * 3
                ):

                    image = residual.reshape(
                        3,
                        32,
                        32,
                    )

                    image = image.transpose(
                        1,
                        2,
                        0,
                    )

                    vmax = np.max(
                        np.abs(image)
                    )

                    if vmax == 0.0:
                        vmax = 1.0

                    display = (
                        image / (2.0 * vmax)
                        +
                        0.5
                    )

                    ax.imshow(
                        np.clip(
                            display,
                            0.0,
                            1.0,
                        )
                    )

                else:

                    ax.plot(
                        residual
                    )

                    ax.axhline(
                        0.0,
                        linewidth=0.8,
                    )

                ax.axis("off")

                if row == 0:
                    ax.set_title(
                        f"Stage {stage}"
                    )

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_recursive_reconstructions(
        self,
        original: np.ndarray,
        reconstructions: list[np.ndarray],
        n_images: int = 8,
        title: str = "Recursive VQ-VAE reconstructions",
    ) -> None:
        """
        Plot the cumulative reconstruction after each stage.
        """
        if len(reconstructions) == 0:
            raise ValueError(
                "At least one reconstruction stage is required."
            )

        if len(original) == 0:
            raise ValueError(
                "Original array must not be empty."
            )

        if n_images <= 0:
            raise ValueError(
                "n_images must be positive."
            )

        n = min(
            n_images,
            len(original),
        )

        for stage, reconstruction in enumerate(
            reconstructions
        ):

            if len(reconstruction) < n:
                raise ValueError(
                    f"Reconstruction stage {stage} "
                    f"contains only {len(reconstruction)} "
                    f"samples, but {n} are required."
                )

            if reconstruction.shape[-1] != (
                original.shape[-1]
            ):
                raise ValueError(
                    f"Reconstruction stage {stage} has "
                    f"dimension {reconstruction.shape[-1]}, "
                    f"but original data have dimension "
                    f"{original.shape[-1]}."
                )

        n_stages = len(
            reconstructions
        )

        fig, axes = plt.subplots(
            n,
            n_stages + 1,
            figsize=(
                3 * (n_stages + 1),
                3 * n,
            ),
            squeeze=False,
        )

        for row in range(n):

            images = [
                original[row]
            ] + [
                reconstruction[row]
                for reconstruction
                in reconstructions
            ]

            for column, image in enumerate(
                images
            ):

                ax = axes[
                    row,
                    column,
                ]

                image = np.asarray(
                    image
                )

                if image.size == 28 * 28:

                    image = image.reshape(
                        28,
                        28,
                    )

                    ax.imshow(
                        np.clip(
                            image,
                            0.0,
                            1.0,
                        ),
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0,
                    )

                elif image.size == (
                    32 * 32 * 3
                ):

                    image = image.reshape(
                        3,
                        32,
                        32,
                    )

                    image = image.transpose(
                        1,
                        2,
                        0,
                    )

                    ax.imshow(
                        np.clip(
                            image,
                            0.0,
                            1.0,
                        )
                    )

                else:

                    ax.plot(
                        image
                    )

                    ax.axhline(
                        0.0,
                        linewidth=0.8,
                    )

                ax.axis("off")

                if row == 0:

                    if column == 0:
                        ax.set_title(
                            "Original"
                        )
                    else:
                        ax.set_title(
                            f"Stage {column - 1}"
                        )

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def _validate_image_plot_arguments(
        self,
        n_images: int,
        n_columns: int,
    ) -> None:
        """
        Validate common image-plotting arguments.
        """
        if n_images <= 0:
            raise ValueError(
                "n_images must be positive."
            )

        if n_columns <= 0:
            raise ValueError(
                "n_columns must be positive."
            )

    @staticmethod
    def _flattened_to_image(
        images: np.ndarray,
    ) -> np.ndarray:
        """
        Backwards-compatible wrapper around ``flattened_to_image``.
        """
        return flattened_to_image(
            images
        )

    @staticmethod
    def _prepare_images(
        images: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Backwards-compatible wrapper around ``prepare_images``.
        """
        return prepare_images(
            images
        )