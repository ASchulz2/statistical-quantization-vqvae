"""
Main entry point for the VQ-VAE experiment.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from checkpointing import save_final_checkpoint
from classes.gif_writer import (
    GifWriter,
    GifWriterSettings,
)
from classes.visualizer import (
    VisualizationSettings,
    Visualizer,
)
from classes.vqvae import VQVAE
from codebook_initialization import initialize_codebook
from data_preprocessing.prepare import prepare_dataset
from evaluation import (
    evaluate_test_set,
    evaluate_validation_set,
)
from experiment_results import (
    create_experiment_result,
    save_experiment_result,
)
from latent_sampling import sample_quantized_latents
from model import create_model
from objective import create_objective
from reproducibility import set_seed
from settings import VQVAEExperimentSettings
from training import create_trainer
from visualization import visualize_results


TRAINING_EXAMPLE_COUNT = 32

VALIDATION_LATENT_GIF_PATH = (
    Path("results")
    /
    "validation_latent_space.gif"
)


def close_figures(
    figures: list[Figure],
) -> None:
    """
    Close all supplied Matplotlib figures.
    """
    for figure in figures:
        plt.close(figure)


def show_figures(
    figures: list[Figure],
) -> None:
    """
    Display collected Matplotlib figures.
    """
    if not figures:
        return

    plt.show(
        block=True
    )

    plt.pause(
        0.001
    )


def _print_experiment_configuration(
    settings: VQVAEExperimentSettings,
    dataset_name: str,
) -> None:
    """Print the active experiment configuration."""
    print()
    print("=" * 70)
    print("VQ-VAE Experiment")
    print("=" * 70)

    print(
        "Device:",
        settings.device,
    )

    print(
        "Dataset:",
        dataset_name.upper(),
    )

    print(
        "EMA mode active:",
        settings.use_ema,
    )

    print(
        "Variance mode active:",
        settings.use_variance,
    )

    print(
        "Latent dimension:",
        settings.latent_dim,
    )

    print(
        "Codebook cardinality:",
        settings.num_embeddings,
    )

    print(
        "Commitment beta:",
        settings.commitment_beta,
    )

    print(
        "Validation fraction:",
        settings.validation_fraction,
    )


def _print_dataset_information(
    train_loader,
    validation_loader,
    test_loader,
) -> int:
    """
    Print dataset information and return the input dimension.
    """
    train_batch = next(
        iter(train_loader)
    )

    if not isinstance(
        train_batch,
        torch.Tensor,
    ):
        raise TypeError(
            "Expected each dataset loader to return "
            "a torch.Tensor."
        )

    input_dim = train_batch.shape[-1]

    print()
    print(
        "Training batch shape:",
        train_batch.shape,
    )

    print(
        "Input dimension:",
        input_dim,
    )

    print(
        "Training samples:",
        len(train_loader.dataset),
    )

    print(
        "Validation samples:",
        len(validation_loader.dataset),
    )

    print(
        "Test samples:",
        len(test_loader.dataset),
    )

    return input_dim


def _collect_training_examples(
    train_loader,
    n_examples: int,
) -> torch.Tensor:
    """
    Collect the first ``n_examples`` from the training loader.
    """
    if n_examples <= 0:
        raise ValueError(
            "n_examples must be positive."
        )

    examples: list[torch.Tensor] = []

    total = 0

    for batch in train_loader:

        if not isinstance(
            batch,
            torch.Tensor,
        ):
            raise TypeError(
                "Expected each dataset loader to return "
                "a torch.Tensor."
            )

        remaining = (
            n_examples
            -
            total
        )

        selected = batch[
            :remaining
        ]

        examples.append(
            selected
        )

        total += len(
            selected
        )

        if total >= n_examples:
            break

    if not examples:
        raise RuntimeError(
            "No training examples were collected."
        )

    return torch.cat(
        examples,
        dim=0,
    )


def _print_evaluation_results(
    metrics: dict,
    split_name: str,
) -> None:
    """
    Print evaluation metrics for a named dataset split.
    """
    print()
    print("=" * 70)
    print(
        f"{split_name.capitalize()} evaluation results"
    )
    print("=" * 70)

    print(
        f"{'Distortion':30s}: "
        f"{metrics['distortion']:.6f}"
    )

    print(
        f"{'Distortion [dB]':30s}: "
        f"{metrics['distortion_db']:.6f}"
    )

    print(
        f"{'RMSE':30s}: "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"{'PSNR':30s}: "
        f"{metrics['psnr']:.6f}"
    )

    print(
        f"{'Rate':30s}: "
        f"{metrics['rate']:.6f}"
    )

    print(
        f"{'Entropy':30s}: "
        f"{metrics['entropy']:.6f}"
    )

    print(
        f"{'Perplexity':30s}: "
        f"{metrics['perplexity']:.6f}"
    )

    print(
        f"{'Utilization':30s}: "
        f"{metrics['utilization']:.6f}"
    )

    print(
        f"{'Active codewords':30s}: "
        f"{metrics['active_codewords']}"
    )

    print(
        f"{'Dead codewords':30s}: "
        f"{metrics['dead_codewords']}"
    )


def _ask_for_test_evaluation() -> bool:
    """
    Ask whether the held-out test set should be evaluated.
    """
    response = input(
        "\nEvaluate the selected model on the held-out "
        "test set? [y/N]: "
    )

    return response.strip().lower() in {
        "y",
        "yes",
    }


def _print_latent_statistics(
    model: VQVAE,
    evaluation_metrics: dict,
    settings: VQVAEExperimentSettings,
    split_name: str,
) -> None:
    """
    Print latent and codeword statistics for a named split.
    """
    latents = evaluation_metrics[
        "latents"
    ]

    print()
    print(
        f"{split_name.capitalize()} codeword histogram:"
    )

    print(
        evaluation_metrics["histogram"].to(
            torch.int64
        )
    )

    print()
    print(
        f"{split_name.capitalize()} encoder latent mean:",
        latents.mean(),
    )

    print(
        f"{split_name.capitalize()} encoder latent std:",
        latents.std(),
    )

    with torch.no_grad():

        evaluation_indices = (
            evaluation_metrics["indices"]
            .to(
                settings.device
            )
        )

        z_q = model.quantizer.quantize(
            evaluation_indices
        )

        z_generating = (
            sample_quantized_latents(
                model=model,
                z_e=latents.to(
                    settings.device
                ),
                noise_scale=(
                    settings.noise_scale
                ),
            )
        )

    print()
    print(
        f"{split_name.capitalize()} deterministic "
        "quantized latent mean:",
        z_q.mean(),
    )

    print(
        f"{split_name.capitalize()} deterministic "
        "quantized latent std:",
        z_q.std(),
    )

    print()
    print(
        f"{split_name.capitalize()} stochastic latent mean:",
        z_generating.mean(),
    )

    print(
        f"{split_name.capitalize()} stochastic latent std:",
        z_generating.std(),
    )


def run_experiment(
    settings: VQVAEExperimentSettings,
) -> None:
    """
    Run one complete VQ-VAE experiment.

    Parameters
    ----------
    settings:
        Complete experiment configuration.
    """
    set_seed(
        settings.seed
    )

    #
    # Dataset
    #

    (
        dataset_name,
        train_loader,
        validation_loader,
        test_loader,
    ) = prepare_dataset(
        settings
    )

    _print_experiment_configuration(
        settings,
        dataset_name,
    )

    input_dim = _print_dataset_information(
        train_loader,
        validation_loader,
        test_loader,
    )

    #
    # Training-set examples.
    #

    training_examples = (
        _collect_training_examples(
            train_loader=train_loader,
            n_examples=TRAINING_EXAMPLE_COUNT,
        )
    )

    #
    # Visualizer.
    #

    visualizer = Visualizer(
        VisualizationSettings()
    )

    training_fig = visualizer.plot_images(
        images=training_examples,
        n_images=TRAINING_EXAMPLE_COUNT,
        n_columns=8,
        title="Training-set examples",
    )

    plt.show(block=True)

    #
    # Model.
    #

    print()
    print("Creating VQ-VAE...")

    model = create_model(
        input_dim=input_dim,
        settings=settings,
    )

    print()
    print(model)

    #
    # Objective.
    #

    objective = create_objective(
        settings
    )

    #
    # Trainer.
    #

    trainer = create_trainer(
        model=model,
        objective=objective,
        settings=settings,
        checkpoint_name=(
            f"vqvae_{dataset_name}_best.pt"
        ),
    )

    #
    # Codebook initialization.
    #

    print()
    print("Initializing codebook...")

    initialize_codebook(
        model=model,
        train_loader=train_loader,
        settings=settings,
    )

    #
    # Training.
    #

    print()
    print("Starting training...")

    start_time = time.perf_counter()

    history = trainer.fit(
        train_loader,
        validation_loader,
    )

    runtime = (
        time.perf_counter()
        -
        start_time
    )

    print()
    print("=" * 70)
    print("Training finished")
    print("=" * 70)

    print(
        f"Runtime: {runtime:.3f} s"
    )

    #
    # Model selection.
    #

    print()
    print("Selected model")
    print("-" * 70)

    print(
        f"{'Criterion':30s}: "
        f"{settings.restore_metric}"
    )

    print(
        f"{'Best validation distortion':30s}: "
        f"{trainer.best_val_distortion:.6f}"
    )

    print(
        f"{'Best distortion epoch':30s}: "
        f"{trainer.best_val_distortion_epoch}"
    )

    print(
        f"{'Best validation loss':30s}: "
        f"{trainer.best_val_loss:.6f}"
    )

    print(
        f"{'Best loss epoch':30s}: "
        f"{trainer.best_val_loss_epoch}"
    )

    print(
        f"{'Selected checkpoint':30s}: "
        f"{trainer.settings.checkpoint_path}"
    )

    #
    # Validation latent GIF.
    #
    # These figures are temporary animation frames and must
    # always be closed after GIF creation.
    #

    try:

        if trainer.validation_latent_figures:

            writer = GifWriter(
                GifWriterSettings(
                    fps=2.0,
                    output_width=1920,
                    output_height=1080,
                    render_dpi=120.0,
                )
            )

            writer.save(
                trainer.validation_latent_figures,
                VALIDATION_LATENT_GIF_PATH,
            )

    finally:

        close_figures(
            trainer.validation_latent_figures[2:-2]
        )

    #
    # Validation evaluation.
    #

    print()
    print(
        "Evaluating selected model on validation set..."
    )

    validation_metrics = evaluate_validation_set(
        model=model,
        validation_loader=validation_loader,
        settings=settings,
    )

    _print_evaluation_results(
        metrics=validation_metrics,
        split_name="validation",
    )

    #
    # Validation visualization.
    #

    print()
    print(
        "Creating validation visualizations..."
    )

    validation_visualization_figures = (
        visualize_results(
            model=model,
            history=history,
            evaluation_metrics=validation_metrics,
            visualizer=visualizer,
            dataset_name=dataset_name,
            settings=settings,
            evaluation_split="validation",
        )
    )

    #
    # Validation diagnostics.
    #

    _print_latent_statistics(
        model=model,
        evaluation_metrics=validation_metrics,
        settings=settings,
        split_name="validation",
    )

    #
    # Optional test evaluation.
    #

    test_metrics = None
    test_visualization_figures: list[Figure] = []

    evaluate_test = _ask_for_test_evaluation()

    if evaluate_test:

        print()
        print(
            "Evaluating selected model on test set..."
        )

        test_metrics = evaluate_test_set(
            model=model,
            test_loader=test_loader,
            settings=settings,
        )

        _print_evaluation_results(
            metrics=test_metrics,
            split_name="test",
        )

        #
        # Test visualization.
        #

        print()
        print(
            "Creating test visualizations..."
        )

        test_visualization_figures = (
            visualize_results(
                model=model,
                history=history,
                evaluation_metrics=test_metrics,
                visualizer=visualizer,
                dataset_name=dataset_name,
                settings=settings,
                evaluation_split="test",
            )
        )

        #
        # Test diagnostics.
        #

        _print_latent_statistics(
            model=model,
            evaluation_metrics=test_metrics,
            settings=settings,
            split_name="test",
        )

    else:

        print()
        print(
            "Test-set evaluation skipped."
        )

    #
    # Persist the experiment result.
    #
    # Test metrics are used when the held-out test set was
    # evaluated. Otherwise, validation metrics are persisted.
    #

    evaluation_metrics = (
        test_metrics
        if test_metrics is not None
        else
        validation_metrics
    )

    evaluation_split = (
        "test"
        if test_metrics is not None
        else
        "validation"
    )

    result = create_experiment_result(
        dataset_name=dataset_name,
        runtime=runtime,
        trainer=trainer,
        evaluation_metrics=evaluation_metrics,
        settings=settings,
    )

    output_path = save_experiment_result(
        result=result,
        settings=settings,
    )

    print()
    print(
        "Experiment result saved:",
        output_path,
    )

    #
    # Final checkpoint.
    #

    final_checkpoint_path = (
        save_final_checkpoint(
            model=model,
            history=history,
            trainer=trainer,
            evaluation_metrics=evaluation_metrics,
            evaluation_split=evaluation_split,
            dataset_name=dataset_name,
            settings=settings,
        )
    )

    print()
    print(
        "Final checkpoint saved:",
        final_checkpoint_path,
    )

    #
    # Display all collected figures.
    #

    figures = [
        training_fig,
        *validation_visualization_figures,
        *test_visualization_figures,
    ]

    show_figures(
        figures
    )

    #
    # Cleanup.
    #

    close_figures(
        figures
    )

    print()
    print("Experiment finished.")


def main() -> None:
    """Run the default VQ-VAE experiment."""
    run_experiment(
        VQVAEExperimentSettings(
            num_embeddings=1024,
            use_ema=True,
            use_variance=True,
            commitment_beta=1.0,
            ema_decay=.99,
            noise_scale=0.0,
            epochs=20,
            plot_validation_latent_space=True,
            train_batch_size=256,
            validation_batch_size=256,
            validation_fraction=.2,
            kmeans_init_batches=4096//64,
            learning_rate=0.001,
            encoder_hidden_dims=(
                512,
                256,
                256,
            ),
            decoder_hidden_dims=(
                256,
                256,
                512,
            ),
            latent_dim=2,
        )
    )


if __name__ == "__main__":
    main()