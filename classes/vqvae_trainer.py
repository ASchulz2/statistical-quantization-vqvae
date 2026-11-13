"""
vqvae_trainer.py
===============

Training, validation and model-selection logic for VQ-VAE.

The trainer separates:

    Training
        - gradient updates
        - EMA codebook updates

    Validation
        - forward pass only
        - no gradient updates
        - no EMA updates

    Test
        - handled externally after model selection

The trainer tracks:

    - training metrics
    - validation metrics
    - best validation loss
    - best validation distortion

It can optionally save the best model according to a
chosen validation criterion.

Part of:

    Toward Statistical Quantization for
    Wake--Sleep Neural Networks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from classes.vqvae import (
    VQVAE,
)

from classes.vqvae_objective import (
    VQVAEObjective,
)

from classes.history import (
    History,
    HistorySettings,
)

from classes.vq_statistics import (
    VQStatistics,
    VQStatisticsSettings,
)

from classes.visualizer import Visualizer
from evaluation import _extract_targets
from matplotlib.figure import Figure


# ============================================================
# Settings
# ============================================================


@dataclass(frozen=True)
class VQVAETrainerSettings:
    """
    Runtime configuration for :class:`VQVAETrainer`.

    Parameters
    ----------
    epochs:
        Number of training epochs.

    learning_rate:
        Learning rate of the Adam optimizer.

    device:
        Device used for training and validation.

    print_every:
        Print training progress every ``print_every`` epochs.

    save_best_checkpoint:
        Whether to save the selected best model checkpoint.

    checkpoint_path:
        Path of the best-model checkpoint.

    restore_metric:
        Validation metric used for model selection.
        Supported values are ``"val_loss"`` and
        ``"val_distortion"``.

    restore_best_model:
        Whether to restore the selected best model after
        training.
    """

    epochs: int = 20

    learning_rate: float = 1e-3

    device: str = "cpu"

    print_every: int = 1

    save_best_checkpoint: bool = False

    checkpoint_path: Path = Path(
        "vqvae_best.pt"
    )

    restore_metric: str = "val_distortion"

    restore_best_model: bool = True

    plot_validation_latent_space: bool = False

    def __post_init__(self) -> None:
        """Validate the trainer configuration."""
        if self.epochs <= 0:
            raise ValueError(
                "epochs must be positive"
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.device not in {
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "device must be either 'cpu' or 'cuda'"
            )

        if self.print_every <= 0:
            raise ValueError(
                "print_every must be positive"
            )

        if self.restore_metric not in {
            "val_loss",
            "val_distortion",
        }:
            raise ValueError(
                "restore_metric must be one of: "
                "'val_loss', 'val_distortion'"
            )


# ============================================================
# Trainer
# ============================================================

class VQVAETrainer:

    """
    Trainer for a VQ-VAE.

    Training:

        model.train()
        forward
        backward
        optimizer.step()

    Validation:

        model.eval()
        forward only

    The test set is deliberately not handled here.

    For EMA-VQ, the quantizer itself checks model.training,
    so validation cannot update the EMA codebook.
    """

    def __init__(
        self,
        model: VQVAE,
        objective: VQVAEObjective,
        settings: VQVAETrainerSettings,
        history: History | None = None,
        visualizer: Visualizer | None = None,
    ):

        self.model = model

        self.objective = objective

        self.settings = settings

        self.visualizer = visualizer

        if (
            settings.plot_validation_latent_space
            and visualizer is None
        ):
            raise ValueError(
                "plot_validation_latent_space requires a Visualizer."
            )

        #
        # Validate selection criterion.
        #

        valid_metrics = {
            "val_loss",
            "val_distortion",
        }


        if settings.restore_metric not in valid_metrics:

            raise ValueError(

                "restore_metric must be one of: "
                "'val_loss', 'val_distortion'."

            )


        #
        # Device
        #

        self.device = torch.device(
            settings.device
        )


        self.model.to(
            self.device
        )


        #
        # Optimizer
        #

        self.optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=settings.learning_rate,

        )


        #
        # History
        #

        if history is None:

            self.history = History(
                HistorySettings()
            )

        else:

            self.history = history


        #
        # Shared VQ statistics
        #

        self.vq_statistics = VQStatistics(

            VQStatisticsSettings()

        )


        #
        # Best validation values
        #

        self.best_val_loss = float(
            "inf"
        )

        self.best_val_distortion = float(
            "inf"
        )


        self.best_val_loss_epoch: int | None = None

        self.best_val_distortion_epoch: int | None = None

        self.validation_latent_figures: list[Figure] = []


        #
        # Separate model states.
        #

        self.best_loss_model_state = None

        self.best_distortion_model_state = None


    # ========================================================
    # Single optimization step
    # ========================================================

    def train_step(
        self,
        batch: torch.Tensor,
    ):

        self.optimizer.zero_grad()


        batch = batch.to(
            self.device
        )


        output = self.model(
            batch
        )


        metrics = self.objective(
            batch,
            output,
        )


        metrics.loss.backward()


        self.optimizer.step()


        return (

            {

                "loss":
                    metrics.loss.item(),

                "reconstruction_loss":
                    metrics.reconstruction_loss.item(),

                "vq_loss":
                    metrics.vq_loss.item(),

                "commitment_loss":
                    metrics.commitment_loss.item(),

                "variance_loss":
                    metrics.variance_loss.item(),

                "distortion":
                    metrics.distortion.item(),

                "rmse":
                    metrics.rmse.item(),

                "psnr":
                    metrics.psnr.item(),

            },

            output.indices.detach(),

        )


    # ========================================================
    # Training epoch
    # ========================================================

    def train_epoch(
        self,
        loader,
    ):

        self.model.train()


        accumulated = {}

        assignments = []

        n_batches = 0


        for batch in loader:

            metrics, indices = self.train_step(
                batch
            )


            #
            # Accumulate scalar metrics.
            #

            for key, value in metrics.items():

                accumulated[key] = (

                    accumulated.get(
                        key,
                        0.0,
                    )

                    +

                    value

                )


            #
            # Store assignments.
            #

            assignments.append(
                indices.cpu()
            )


            n_batches += 1


        if n_batches == 0:

            raise RuntimeError(
                "Training loader produced no batches."
            )


        #
        # Average batch metrics.
        #

        for key in accumulated:

            accumulated[key] /= n_batches


        #
        # Global training codebook statistics.
        #

        assignments = torch.cat(
            assignments
        )


        vq_stats = self.vq_statistics.analyze(

            assignments.numpy(),

            self.model.quantizer.settings.num_embeddings,

        )


        accumulated.update(

            {

                "perplexity":
                    vq_stats["perplexity"],

                "utilization":
                    vq_stats["utilization"],

                "entropy":
                    vq_stats["entropy"],

                "dead_codewords":
                    vq_stats["dead_codewords"],

            }

        )


        return accumulated


    # ========================================================
    # Validation epoch
    # ========================================================

    @torch.no_grad()
    def validate(
        self,
        loader,
    ):

        """
        Evaluate the model on validation data.

        No gradients are computed and no parameters are
        modified.

        For EMA quantization, model.eval() also prevents
        EMA codebook updates.
        """

        self.model.eval()


        accumulated = {}

        assignments = []

        n_batches = 0


        for batch in loader:

            batch = batch.to(
                self.device
            )


            output = self.model(
                batch
            )


            metrics = self.objective(

                batch,

                output,

            )


            values = {

                "loss":
                    metrics.loss.item(),

                "reconstruction_loss":
                    metrics.reconstruction_loss.item(),

                "vq_loss":
                    metrics.vq_loss.item(),

                "commitment_loss":
                    metrics.commitment_loss.item(),

                "variance_loss":
                    metrics.variance_loss.item(),

                "distortion":
                    metrics.distortion.item(),

                "rmse":
                    metrics.rmse.item(),

                "psnr":
                    metrics.psnr.item(),

            }


            for key, value in values.items():

                accumulated[key] = (

                    accumulated.get(
                        key,
                        0.0,
                    )

                    +

                    value

                )


            assignments.append(
                output.indices.cpu()
            )


            n_batches += 1


        if n_batches == 0:

            raise RuntimeError(
                "Validation loader produced no batches."
            )


        #
        # Average validation metrics.
        #

        for key in accumulated:

            accumulated[key] /= n_batches


        #
        # Global validation codebook statistics.
        #

        assignments = torch.cat(
            assignments
        )


        vq_stats = self.vq_statistics.analyze(

            assignments.numpy(),

            self.model.quantizer.settings.num_embeddings,

        )


        accumulated.update(

            {

                "perplexity":
                    vq_stats["perplexity"],

                "utilization":
                    vq_stats["utilization"],

                "entropy":
                    vq_stats["entropy"],

                "dead_codewords":
                    vq_stats["dead_codewords"],

            }

        )


        return accumulated


    # ========================================================
    # Capture model state
    # ========================================================

    def _capture_model_state(
        self,
    ) -> dict[str, torch.Tensor]:

        """
        Store a CPU copy of the current model state.

        Keeping the state on CPU avoids unnecessarily retaining
        GPU memory during long training runs.
        """

        return {

            key:
                value.detach()
                .cpu()
                .clone()

            for key, value
            in self.model.state_dict().items()

        }


    # ========================================================
    # Restore model state
    # ========================================================

    def _restore_model_state(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:

        restored = {

            key:
                value.to(
                    self.device
                )

            for key, value
            in state.items()

        }


        self.model.load_state_dict(
            restored
        )


        # ========================================================
    # Update best model states
    # ========================================================

    def _update_best_model_states(
        self,
        epoch: int,
        validation_metrics: dict,
    ) -> bool:
        """
        Update the stored best model states.

        Both validation loss and validation distortion are tracked
        independently. The return value indicates whether the
        validation metric configured by ``restore_metric`` improved
        during the current epoch.

        Parameters
        ----------
        epoch:
            Current training epoch.

        validation_metrics:
            Validation metrics produced by :meth:`validate`.

        Returns
        -------
        bool
            ``True`` if the configured model-selection criterion
            improved, otherwise ``False``.
        """
        current_val_loss = validation_metrics[
            "loss"
        ]

        current_val_distortion = validation_metrics[
            "distortion"
        ]

        #
        # Best validation loss.
        #

        loss_improved = (
            current_val_loss
            <
            self.best_val_loss
        )

        if loss_improved:

            self.best_val_loss = (
                current_val_loss
            )

            self.best_val_loss_epoch = (
                epoch
            )

            self.best_loss_model_state = (
                self._capture_model_state()
            )

        #
        # Best validation distortion.
        #

        distortion_improved = (
            current_val_distortion
            <
            self.best_val_distortion
        )

        if distortion_improved:

            self.best_val_distortion = (
                current_val_distortion
            )

            self.best_val_distortion_epoch = (
                epoch
            )

            self.best_distortion_model_state = (
                self._capture_model_state()
            )

        #
        # Selected criterion.
        #

        if self.settings.restore_metric == "val_loss":

            return loss_improved

        return distortion_improved


    # ========================================================
    # Save best checkpoint
    # ========================================================

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict,
        validation_metrics: dict,
        selection_metric: str,
    ) -> None:

        if not self.settings.save_best_checkpoint:

            return


        path = Path(
            self.settings.checkpoint_path
        )


        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        checkpoint = {

            "epoch":
                epoch,

            "selection_metric":
                selection_metric,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "train_metrics":
                train_metrics,

            "validation_metrics":
                validation_metrics,

            "history":
                self.history.as_dict(), # type: ignore

            "best_val_loss":
                self.best_val_loss,

            "best_val_distortion":
                self.best_val_distortion,

            "best_val_loss_epoch":
                self.best_val_loss_epoch,

            "best_val_distortion_epoch":
                self.best_val_distortion_epoch,

        }


        torch.save(
            checkpoint,
            path,
        )


    # ========================================================
    # Restore selected best model
    # ========================================================

    def restore_best(
        self,
    ) -> None:

        if self.settings.restore_metric == "val_loss":

            state = (
                self.best_loss_model_state
            )

        else:

            state = (
                self.best_distortion_model_state
            )


        if state is None:

            return


        self._restore_model_state(
            state
        )


    # ========================================================
    # Fit
    # ========================================================

    def fit(
        self,
        train_loader,
        validation_loader=None,
    ):
        """
        Train the VQ-VAE and optionally perform validation-based
        model selection.

        Parameters
        ----------
        train_loader:
            Training data loader.

        validation_loader:
            Optional validation data loader.

        Returns
        -------
        History
            Training and validation history.
        """
        if (
            validation_loader is not None
            and self.settings.plot_validation_latent_space
        ):
            self._plot_validation_latent_space(
                validation_loader,
                epoch=None,
            )

        for epoch in range(
            self.settings.epochs
        ):

            epoch_number = epoch + 1


            # ==================================================
            # Training
            # ==================================================

            train_metrics = self.train_epoch(
                train_loader
            )


            # ==================================================
            # Validation
            # ==================================================

            validation_metrics = None

            if validation_loader is not None:

                validation_metrics = self.validate(
                    validation_loader
                )
                self._plot_validation_latent_space(
                    validation_loader,
                    epoch=epoch_number,
                )             


            # ==================================================
            # History
            # ==================================================

            history_metrics = {

                key:
                    value

                for key, value
                in train_metrics.items()

            }


            if validation_metrics is not None:

                history_metrics.update(

                    {

                        f"val_{key}":
                            value

                        for key, value
                        in validation_metrics.items()

                    }

                )


            self.history.update(
                history_metrics
            )


            # ==================================================
            # Model selection
            # ==================================================

            if validation_metrics is not None:

                selected_improved = (
                    self._update_best_model_states(
                        epoch=epoch_number,
                        validation_metrics=validation_metrics,
                    )
                )

                if selected_improved:

                    self._save_checkpoint(
                        epoch=epoch_number,
                        train_metrics=train_metrics,
                        validation_metrics=validation_metrics,
                        selection_metric=(
                            self.settings.restore_metric
                        ),
                    )


            # ==================================================
            # Console output
            # ==================================================

            if (

                epoch
                %
                self.settings.print_every
                ==
                0

            ):

                message = (

                    f"Epoch {epoch_number:03d}"

                    f" | Train Loss="
                    f"{train_metrics['loss']:.5f}"

                    f" | Train Dist="
                    f"{train_metrics['distortion']:.5f}"

                    f" | Train Perp="
                    f"{train_metrics['perplexity']:.3f}"

                    f" | Train Util="
                    f"{train_metrics['utilization']:.3f}"

                )


                if validation_metrics is not None:

                    message += (

                        f" | Val Loss="
                        f"{validation_metrics['loss']:.5f}"

                        f" | Val Dist="
                        f"{validation_metrics['distortion']:.5f}"

                        f" | Val Perp="
                        f"{validation_metrics['perplexity']:.3f}"

                        f" | Val Util="
                        f"{validation_metrics['utilization']:.3f}"

                    )


                print(
                    message
                )


        # ======================================================
        # Restore best selected model
        # ======================================================

        if (

            validation_loader is not None

            and

            self.settings.restore_best_model

        ):

            self.restore_best()


        # ======================================================
        # Return history
        # ======================================================

        return self.history

    @torch.no_grad()
    def _collect_validation_latents(
        self,
        loader,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Collect the complete validation-set encoder representations
        and, when available, their targets.
        """
        self.model.eval()

        latents = []

        for batch in loader:
            batch = batch.to(self.device)
            z_e = self.model.encode(batch)
            latents.append(z_e.detach().cpu())

        if not latents:
            raise RuntimeError(
                "Validation loader produced no batches."
            )

        latent_tensor = torch.cat(
            latents,
            dim=0,
        )

        dataset = getattr(
            loader,
            "dataset",
            None,
        )

        targets = _extract_targets(dataset)

        if targets is not None:
            targets = targets.detach().cpu()

            if targets.shape[0] != latent_tensor.shape[0]:
                raise RuntimeError(
                    "Validation targets and latent representations "
                    "have different numbers of samples."
                )

        return latent_tensor, targets

    def _plot_validation_latent_space(
        self,
        validation_loader,
        *,
        epoch: int | None,
    ) -> None:
        """
        Plot the complete validation-set latent representation.
        """
        if not self.settings.plot_validation_latent_space:
            return

        if self.visualizer is None:
            raise RuntimeError(
                "Validation latent-space visualization requires "
                "a Visualizer."
            )

        embedding_dim = (
            self.model.quantizer.settings.embedding_dim
        )

        if embedding_dim != 2:
            return

        latents, targets = (
            self._collect_validation_latents(
                validation_loader
            )
        )

        codewords = (
            self.model.quantizer.embedding_mean.weight
            .detach()
            .cpu()
            .numpy()
        )

        title = (
            "Validation latent space — initial"
            if epoch is None
            else
            f"Validation latent space — epoch {epoch}"
        )

        self.validation_latent_figures.append(
            self.visualizer.plot_latent_space(
                latents.numpy(),
                codewords=codewords,
                targets=(
                    None
                    if targets is None
                    else targets.numpy()
                ),
                title=title,
            )
        )