
"""
Tests for the VQ-VAE experiment entry point.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from matplotlib.figure import Figure

import main as main_module
from settings import VQVAEExperimentSettings


class DummyLoader:
    """Minimal loader used by experiment orchestration tests."""

    def __init__(
        self,
        batch_size: int = 4,
        input_dim: int = 8,
        n_samples: int = 16,
    ) -> None:
        self.batch = torch.zeros(
            batch_size,
            input_dim,
        )

        self.n_samples = n_samples

    def __iter__(self):
        yield self.batch

    def __len__(
        self,
    ) -> int:
        return 1

    @property
    def dataset(self):
        return list(
            range(self.n_samples)
        )


class DummyHistory:
    """Minimal history object."""

    def as_dict(
        self,
    ) -> dict:
        return {
            "loss": [1.0],
            "val_loss": [1.1],
        }


class DummyModel:
    """Minimal model placeholder."""

    pass


class DummyTrainer:
    """Minimal trainer used by experiment orchestration tests."""

    def __init__(
        self,
        history,
        validation_latent_figures=None,
    ) -> None:
        self.history = history

        self.validation_latent_figures = (
            validation_latent_figures
            if validation_latent_figures is not None
            else []
        )

        self.best_val_loss = 1.0
        self.best_val_distortion = 0.5
        self.best_val_loss_epoch = 1
        self.best_val_distortion_epoch = 1

        self.settings = SimpleNamespace(
            checkpoint_path=Path(
                "checkpoints/test_best.pt"
            )
        )

        self.fit_called = False

    def fit(
        self,
        train_loader,
        validation_loader,
    ):
        self.fit_called = True

        return self.history


def create_test_metrics() -> dict:
    """
    Create the metrics expected by evaluation-dependent
    experiment components.
    """
    return {
        "originals": torch.zeros(
            4,
            8,
        ),
        "reconstructions": torch.zeros(
            4,
            8,
        ),
        "indices": torch.zeros(
            4,
            dtype=torch.long,
        ),
        "latents": torch.zeros(
            4,
            2,
        ),
        "histogram": torch.tensor(
            [4, 0, 0, 0]
        ),
        "distortion": 0.10,
        "distortion_db": 10.0,
        "rmse": 0.316,
        "psnr": 10.0,
        "rate": 0.0,
        "entropy": 0.0,
        "perplexity": 1.0,
        "utilization": 0.25,
        "active_codewords": 1,
        "dead_codewords": 3,
    }


def test_run_experiment_orchestrates_pipeline_with_test_evaluation(
    monkeypatch,
) -> None:
    calls: list[str] = []
    shown_figures: list[list[Figure]] = []

    expected_settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=2,
        num_embeddings=4,
        plot_validation_latent_space=True,
        checkpoint_directory=Path(
            "checkpoints"
        ),
        results_path=Path(
            "results/test.csv"
        ),
    )

    expected_train_loader = DummyLoader()
    expected_validation_loader = DummyLoader()
    expected_test_loader = DummyLoader()

    expected_history = DummyHistory()

    expected_trainer = DummyTrainer(
        history=expected_history,
    )

    expected_model = DummyModel()

    expected_validation_metrics = (
        create_test_metrics()
    )

    expected_test_metrics = (
        create_test_metrics()
    )

    training_figure = Figure()
    validation_visualization_figure = Figure()
    test_visualization_figure = Figure()

    validation_visualization_figures = [
        validation_visualization_figure,
    ]

    test_visualization_figures = [
        test_visualization_figure,
    ]

    def fake_set_seed(
        seed: int,
    ) -> None:
        calls.append(
            f"set_seed:{seed}"
        )

        assert seed == expected_settings.seed

    def fake_prepare_dataset(
        settings,
    ):
        calls.append(
            "prepare_dataset"
        )

        assert settings is expected_settings

        return (
            "mnist",
            expected_train_loader,
            expected_validation_loader,
            expected_test_loader,
        )

    def fake_create_model(
        input_dim,
        settings,
    ):
        calls.append(
            "create_model"
        )

        assert input_dim == 8
        assert settings is expected_settings

        return expected_model

    def fake_create_objective(
        settings,
    ):
        calls.append(
            "create_objective"
        )

        assert settings is expected_settings

        return object()

    def fake_create_trainer(
        model,
        objective,
        settings,
        checkpoint_name=None,
        history=None,
    ):
        calls.append(
            "create_trainer"
        )

        assert model is expected_model
        assert settings is expected_settings

        assert checkpoint_name == (
            "vqvae_mnist_best.pt"
        )

        assert history is None

        return expected_trainer

    def fake_initialize_codebook(
        model,
        train_loader,
        settings,
    ):
        calls.append(
            "initialize_codebook"
        )

        assert model is expected_model
        assert train_loader is expected_train_loader
        assert settings is expected_settings

    def fake_evaluate_validation_set(
        model,
        validation_loader,
        settings,
    ):
        calls.append(
            "evaluate_validation_set"
        )

        assert model is expected_model
        assert validation_loader is (
            expected_validation_loader
        )
        assert settings is expected_settings

        return expected_validation_metrics

    def fake_evaluate_test_set(
        model,
        test_loader,
        settings,
    ):
        calls.append(
            "evaluate_test_set"
        )

        assert model is expected_model
        assert test_loader is expected_test_loader
        assert settings is expected_settings

        return expected_test_metrics

    def fake_ask_for_test_evaluation():
        calls.append(
            "ask_for_test_evaluation"
        )

        return True

    def fake_create_result(
        dataset_name,
        runtime,
        trainer,
        evaluation_metrics,
        settings,
    ):
        calls.append(
            "create_experiment_result"
        )

        assert dataset_name == "mnist"
        assert trainer is expected_trainer
        assert evaluation_metrics is (
            expected_test_metrics
        )
        assert settings is expected_settings
        assert runtime >= 0.0

        return object()

    def fake_save_result(
        result,
        settings,
    ):
        calls.append(
            "save_experiment_result"
        )

        assert settings is expected_settings

        return Path(
            "results/test.csv"
        )

    def fake_visualize_results(
        model,
        history,
        evaluation_metrics,
        visualizer,
        dataset_name,
        settings,
        evaluation_split,
    ):
        calls.append(
            "visualize_results"
        )

        assert model is expected_model
        assert history is expected_history
        assert dataset_name == "mnist"
        assert settings is expected_settings

        if evaluation_split == "validation":
            assert evaluation_metrics is (
                expected_validation_metrics
            )

            return validation_visualization_figures

        if evaluation_split == "test":
            assert evaluation_metrics is (
                expected_test_metrics
            )

            return test_visualization_figures

        raise AssertionError(
            f"Unexpected split: {evaluation_split}"
        )

    def fake_save_final_checkpoint(
        model,
        history,
        trainer,
        evaluation_metrics,
        evaluation_split,
        dataset_name,
        settings,
    ):
        calls.append(
            "save_final_checkpoint"
        )

        assert model is expected_model
        assert history is expected_history
        assert trainer is expected_trainer
        assert evaluation_metrics is (
            expected_test_metrics
        )
        assert evaluation_split == "test"
        assert dataset_name == "mnist"
        assert settings is expected_settings

        return Path(
            "checkpoints/vqvae_mnist.pt"
        )

    def fake_print_latent_statistics(
        model,
        evaluation_metrics,
        settings,
        split_name,
    ):
        calls.append(
            "print_latent_statistics"
        )

        assert model is expected_model
        assert settings is expected_settings

        if split_name == "validation":
            assert evaluation_metrics is (
                expected_validation_metrics
            )

        elif split_name == "test":
            assert evaluation_metrics is (
                expected_test_metrics
            )

        else:
            raise AssertionError(
                f"Unexpected split: {split_name}"
            )

    def fake_visualizer(
        settings,
    ):
        calls.append(
            "visualizer"
        )

        class FakeVisualizer:
            def plot_images(
                self,
                images,
                n_images,
                n_columns,
                title,
            ):
                calls.append(
                    "plot_images"
                )

                assert n_images == 32
                assert n_columns == 8
                assert title == (
                    "Training-set examples"
                )

                return training_figure

        return FakeVisualizer()

    def fake_show_figures(
        figures,
    ):
        calls.append(
            "show_figures"
        )

        shown_figures.append(
            figures
        )

    def fake_close_figures(
        figures,
    ):
        calls.append(
            "close_figures"
        )

    monkeypatch.setattr(
        main_module,
        "set_seed",
        fake_set_seed,
    )

    monkeypatch.setattr(
        main_module,
        "prepare_dataset",
        fake_prepare_dataset,
    )

    monkeypatch.setattr(
        main_module,
        "create_model",
        fake_create_model,
    )

    monkeypatch.setattr(
        main_module,
        "create_objective",
        fake_create_objective,
    )

    monkeypatch.setattr(
        main_module,
        "create_trainer",
        fake_create_trainer,
    )

    monkeypatch.setattr(
        main_module,
        "initialize_codebook",
        fake_initialize_codebook,
    )

    monkeypatch.setattr(
        main_module,
        "evaluate_validation_set",
        fake_evaluate_validation_set,
    )

    monkeypatch.setattr(
        main_module,
        "evaluate_test_set",
        fake_evaluate_test_set,
    )

    monkeypatch.setattr(
        main_module,
        "_ask_for_test_evaluation",
        fake_ask_for_test_evaluation,
    )

    monkeypatch.setattr(
        main_module,
        "create_experiment_result",
        fake_create_result,
    )

    monkeypatch.setattr(
        main_module,
        "save_experiment_result",
        fake_save_result,
    )

    monkeypatch.setattr(
        main_module,
        "visualize_results",
        fake_visualize_results,
    )

    monkeypatch.setattr(
        main_module,
        "save_final_checkpoint",
        fake_save_final_checkpoint,
    )

    monkeypatch.setattr(
        main_module,
        "_print_latent_statistics",
        fake_print_latent_statistics,
    )

    monkeypatch.setattr(
        main_module,
        "Visualizer",
        fake_visualizer,
    )

    monkeypatch.setattr(
        main_module,
        "VisualizationSettings",
        lambda: object(),
    )

    monkeypatch.setattr(
        main_module,
        "show_figures",
        fake_show_figures,
    )

    monkeypatch.setattr(
        main_module,
        "close_figures",
        fake_close_figures,
    )

    main_module.run_experiment(
        expected_settings
    )

    assert expected_trainer.fit_called is True

    assert calls == [
        "set_seed:42",
        "prepare_dataset",
        "visualizer",
        "plot_images",
        "create_model",
        "create_objective",
        "create_trainer",
        "initialize_codebook",
        "close_figures",
        "evaluate_validation_set",
        "visualize_results",
        "print_latent_statistics",
        "ask_for_test_evaluation",
        "evaluate_test_set",
        "visualize_results",
        "print_latent_statistics",
        "create_experiment_result",
        "save_experiment_result",
        "save_final_checkpoint",
        "show_figures",
        "close_figures",
    ]

    assert len(
        shown_figures
    ) == 1

    displayed_figures = (
        shown_figures[0]
    )

    assert displayed_figures == [
        training_figure,
        validation_visualization_figure,
        test_visualization_figure,
    ]


def test_run_experiment_skips_test_evaluation(
    monkeypatch,
) -> None:
    expected_settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=2,
        num_embeddings=4,
        plot_validation_latent_space=False,
        checkpoint_directory=Path(
            "checkpoints"
        ),
        results_path=Path(
            "results/test.csv"
        ),
    )

    expected_train_loader = DummyLoader()
    expected_validation_loader = DummyLoader()
    expected_test_loader = DummyLoader()

    expected_history = DummyHistory()

    expected_trainer = DummyTrainer(
        history=expected_history,
    )

    expected_model = DummyModel()

    expected_validation_metrics = (
        create_test_metrics()
    )

    expected_test_metrics = (
        create_test_metrics()
    )

    training_figure = Figure()
    validation_figure = Figure()

    evaluate_validation_set = Mock(
        return_value=expected_validation_metrics
    )

    evaluate_test_set = Mock(
        return_value=expected_test_metrics
    )

    ask_for_test_evaluation = Mock(
        return_value=False
    )

    create_experiment_result = Mock(
        return_value=object()
    )

    save_experiment_result = Mock(
        return_value=Path(
            "results/test.csv"
        )
    )

    visualize_results = Mock(
        return_value=[
            validation_figure,
        ]
    )

    print_latent_statistics = Mock()

    save_final_checkpoint = Mock(
        return_value=Path(
            "checkpoints/vqvae_mnist.pt"
        )
    )

    show_figures = Mock()
    close_figures = Mock()

    class FakeVisualizer:
        def plot_images(
            self,
            images,
            n_images,
            n_columns,
            title,
        ):
            return training_figure

    monkeypatch.setattr(
        main_module,
        "prepare_dataset",
        Mock(
            return_value=(
                "mnist",
                expected_train_loader,
                expected_validation_loader,
                expected_test_loader,
            )
        ),
    )

    monkeypatch.setattr(
        main_module,
        "Visualizer",
        lambda settings: FakeVisualizer(),
    )

    monkeypatch.setattr(
        main_module,
        "VisualizationSettings",
        lambda: object(),
    )

    monkeypatch.setattr(
        main_module,
        "create_model",
        Mock(
            return_value=expected_model
        ),
    )

    monkeypatch.setattr(
        main_module,
        "create_objective",
        Mock(
            return_value=object()
        ),
    )

    monkeypatch.setattr(
        main_module,
        "create_trainer",
        Mock(
            return_value=expected_trainer
        ),
    )

    monkeypatch.setattr(
        main_module,
        "initialize_codebook",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "evaluate_validation_set",
        evaluate_validation_set,
    )

    monkeypatch.setattr(
        main_module,
        "evaluate_test_set",
        evaluate_test_set,
    )

    monkeypatch.setattr(
        main_module,
        "_ask_for_test_evaluation",
        ask_for_test_evaluation,
    )

    monkeypatch.setattr(
        main_module,
        "create_experiment_result",
        create_experiment_result,
    )

    monkeypatch.setattr(
        main_module,
        "save_experiment_result",
        save_experiment_result,
    )

    monkeypatch.setattr(
        main_module,
        "visualize_results",
        visualize_results,
    )

    monkeypatch.setattr(
        main_module,
        "_print_latent_statistics",
        print_latent_statistics,
    )

    monkeypatch.setattr(
        main_module,
        "save_final_checkpoint",
        save_final_checkpoint,
    )

    monkeypatch.setattr(
        main_module,
        "show_figures",
        show_figures,
    )

    monkeypatch.setattr(
        main_module,
        "close_figures",
        close_figures,
    )

    main_module.run_experiment(
        expected_settings
    )

    evaluate_validation_set.assert_called_once_with(
        model=expected_model,
        validation_loader=expected_validation_loader,
        settings=expected_settings,
    )

    evaluate_test_set.assert_not_called()

    ask_for_test_evaluation.assert_called_once_with()

    visualize_results.assert_called_once_with(
        model=expected_model,
        history=expected_history,
        evaluation_metrics=expected_validation_metrics,
        visualizer=(
            visualize_results.call_args.kwargs["visualizer"]
        ),
        dataset_name="mnist",
        settings=expected_settings,
        evaluation_split="validation",
    )

    print_latent_statistics.assert_called_once_with(
        model=expected_model,
        evaluation_metrics=expected_validation_metrics,
        settings=expected_settings,
        split_name="validation",
    )

    create_experiment_result.assert_called_once_with(
        dataset_name="mnist",
        runtime=(
            create_experiment_result.call_args.kwargs[
                "runtime"
            ]
        ),
        trainer=expected_trainer,
        evaluation_metrics=expected_validation_metrics,
        settings=expected_settings,
    )

    save_experiment_result.assert_called_once_with(
        result=(
            create_experiment_result.return_value
        ),
        settings=expected_settings,
    )

    save_final_checkpoint.assert_called_once_with(
        model=expected_model,
        history=expected_history,
        trainer=expected_trainer,
        evaluation_metrics=expected_validation_metrics,
        evaluation_split="validation",
        dataset_name="mnist",
        settings=expected_settings,
    )

    show_figures.assert_called_once_with(
        [
            training_figure,
            validation_figure,
        ]
    )

    close_figures.assert_called()

    assert (
        close_figures.call_count
        == 2
    )


def test_main_creates_default_settings_and_runs_experiment(
    monkeypatch,
) -> None:
    expected_settings = (
        VQVAEExperimentSettings(
            device="cpu",
        )
    )

    captured: dict[str, object] = {}

    def fake_run_experiment(
        received_settings,
    ) -> None:
        captured["settings"] = (
            received_settings
        )

    monkeypatch.setattr(
        main_module,
        "run_experiment",
        fake_run_experiment,
    )

    captured_constructor_kwargs: dict[
        str,
        object,
    ] = {}

    def fake_settings_factory(
        **kwargs,
    ):
        captured_constructor_kwargs.update(
            kwargs
        )

        return expected_settings

    monkeypatch.setattr(
        main_module,
        "VQVAEExperimentSettings",
        fake_settings_factory,
    )

    main_module.main()

    assert (
        captured["settings"]
        is expected_settings
    )

    assert captured_constructor_kwargs == {
        "num_embeddings": 1024,
        "use_ema": True,
        "use_variance": True,
        "commitment_beta": 1.0,
        "ema_decay": .99,
        "noise_scale": 0.0,
        "epochs": 20,
        "plot_validation_latent_space": True,
        "train_batch_size": 256,
        "validation_batch_size": 256,
        "validation_fraction": .2,
        "kmeans_init_batches": 4096//64,
        "learning_rate": 0.001,
        "encoder_hidden_dims": (
            512,
            256,
            256,
        ),
        "decoder_hidden_dims": (
            256,
            256,
            512,
        ),
        "latent_dim": 2,
    }


def test_show_figures_displays_collected_figures(
    monkeypatch,
) -> None:
    figures = [
        Mock(),
        Mock(),
        Mock(),
    ]

    show_calls: list[tuple] = []
    pause_calls: list[float] = []

    monkeypatch.setattr(
        main_module.plt,
        "show",
        lambda *args, **kwargs: show_calls.append(
            (
                args,
                kwargs,
            )
        ),
    )

    monkeypatch.setattr(
        main_module.plt,
        "pause",
        lambda interval: pause_calls.append(
            interval
        ),
    )

    main_module.show_figures(
        figures
    )

    for figure in figures:
        figure.show.assert_not_called()

    assert show_calls == [
        (
            (),
            {
                "block": True,
            },
        ),
    ]

    assert pause_calls == [
        0.001,
    ]




def test_close_figures_closes_all_figures(
    monkeypatch,
) -> None:
    figures = [
        Figure(),
        Figure(),
        Figure(),
    ]

    close_calls: list[Figure] = []

    monkeypatch.setattr(
        main_module.plt,
        "close",
        lambda figure: close_calls.append(
            figure
        ),
    )

    main_module.close_figures(
        figures
    )

    assert close_calls == figures


def test_ask_for_test_evaluation_accepts_yes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "y",
    )

    assert (
        main_module._ask_for_test_evaluation()
        is True
    )


def test_ask_for_test_evaluation_defaults_to_no(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "",
    )

    assert (
        main_module._ask_for_test_evaluation()
        is False
    )


def test_run_experiment_skips_empty_validation_latent_gif(
    monkeypatch,
) -> None:
    expected_settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=2,
        num_embeddings=4,
        plot_validation_latent_space=False,
        checkpoint_directory=Path(
            "checkpoints"
        ),
        results_path=Path(
            "results/test.csv"
        ),
    )

    expected_train_loader = DummyLoader()
    expected_validation_loader = DummyLoader()
    expected_test_loader = DummyLoader()

    expected_history = DummyHistory()

    expected_trainer = DummyTrainer(
        history=expected_history,
        validation_latent_figures=[],
    )

    expected_model = DummyModel()

    expected_validation_metrics = (
        create_test_metrics()
    )

    training_figure = Figure()

    writer_class = Mock()

    visualize_results = Mock(
        return_value=[]
    )

    evaluate_validation_set = Mock(
        return_value=expected_validation_metrics
    )

    evaluate_test_set = Mock()

    ask_for_test_evaluation = Mock(
        return_value=False
    )

    show_figures = Mock()
    close_figures = Mock()

    class FakeVisualizer:
        def plot_images(
            self,
            images,
            n_images,
            n_columns,
            title,
        ):
            return training_figure

    monkeypatch.setattr(
        main_module,
        "prepare_dataset",
        Mock(
            return_value=(
                "mnist",
                expected_train_loader,
                expected_validation_loader,
                expected_test_loader,
            )
        ),
    )

    monkeypatch.setattr(
        main_module,
        "Visualizer",
        lambda settings: FakeVisualizer(),
    )

    monkeypatch.setattr(
        main_module,
        "VisualizationSettings",
        lambda: object(),
    )

    monkeypatch.setattr(
        main_module,
        "create_model",
        Mock(
            return_value=expected_model
        ),
    )

    monkeypatch.setattr(
        main_module,
        "create_objective",
        Mock(
            return_value=object()
        ),
    )

    monkeypatch.setattr(
        main_module,
        "create_trainer",
        Mock(
            return_value=expected_trainer
        ),
    )

    monkeypatch.setattr(
        main_module,
        "initialize_codebook",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "evaluate_validation_set",
        evaluate_validation_set,
    )

    monkeypatch.setattr(
        main_module,
        "evaluate_test_set",
        evaluate_test_set,
    )

    monkeypatch.setattr(
        main_module,
        "_ask_for_test_evaluation",
        ask_for_test_evaluation,
    )

    monkeypatch.setattr(
        main_module,
        "create_experiment_result",
        Mock(
            return_value=object()
        ),
    )

    monkeypatch.setattr(
        main_module,
        "save_experiment_result",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "visualize_results",
        visualize_results,
    )

    monkeypatch.setattr(
        main_module,
        "_print_latent_statistics",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "save_final_checkpoint",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "GifWriter",
        writer_class,
    )

    monkeypatch.setattr(
        main_module,
        "show_figures",
        show_figures,
    )

    monkeypatch.setattr(
        main_module,
        "close_figures",
        close_figures,
    )

    main_module.run_experiment(
        expected_settings
    )

    writer_class.assert_not_called()

    evaluate_validation_set.assert_called_once()

    evaluate_test_set.assert_not_called()

    ask_for_test_evaluation.assert_called_once_with()

    visualize_results.assert_called_once()

    show_figures.assert_called_once_with(
        [training_figure]
    )

    assert (
        close_figures.call_count
        == 2
    )


def test_run_experiment_saves_validation_latent_gif(
    monkeypatch,
) -> None:
    validation_figures = [
        Figure(),
        Figure(),
        Figure(),
        Figure(),
    ]

    expected_settings = VQVAEExperimentSettings(
        device="cpu",
        latent_dim=2,
        num_embeddings=4,
        plot_validation_latent_space=True,
        checkpoint_directory=Path(
            "checkpoints"
        ),
        results_path=Path(
            "results/test.csv"
        ),
    )

    expected_train_loader = DummyLoader()
    expected_validation_loader = DummyLoader()
    expected_test_loader = DummyLoader()

    expected_history = DummyHistory()
    expected_model = DummyModel()

    expected_validation_metrics = (
        create_test_metrics()
    )

    expected_trainer = DummyTrainer(
        history=expected_history,
        validation_latent_figures=(
            validation_figures
        ),
    )

    writer = Mock()
    writer_class = Mock(
        return_value=writer
    )

    training_figure = Figure()

    visualize_results = Mock(
        return_value=[]
    )

    evaluate_validation_set = Mock(
        return_value=expected_validation_metrics
    )

    ask_for_test_evaluation = Mock(
        return_value=False
    )

    show_figures = Mock()
    close_figures = Mock()

    class FakeVisualizer:
        def plot_images(
            self,
            images,
            n_images,
            n_columns,
            title,
        ):
            return training_figure

    monkeypatch.setattr(
        main_module,
        "prepare_dataset",
        Mock(
            return_value=(
                "mnist",
                expected_train_loader,
                expected_validation_loader,
                expected_test_loader,
            )
        ),
    )

    monkeypatch.setattr(
        main_module,
        "Visualizer",
        lambda settings: FakeVisualizer(),
    )

    monkeypatch.setattr(
        main_module,
        "VisualizationSettings",
        lambda: object(),
    )

    monkeypatch.setattr(
        main_module,
        "create_model",
        Mock(
            return_value=expected_model
        ),
    )

    monkeypatch.setattr(
        main_module,
        "create_objective",
        Mock(
            return_value=object()
        ),
    )

    monkeypatch.setattr(
        main_module,
        "create_trainer",
        Mock(
            return_value=expected_trainer
        ),
    )

    monkeypatch.setattr(
        main_module,
        "initialize_codebook",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "evaluate_validation_set",
        evaluate_validation_set,
    )

    monkeypatch.setattr(
        main_module,
        "evaluate_test_set",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "_ask_for_test_evaluation",
        ask_for_test_evaluation,
    )

    monkeypatch.setattr(
        main_module,
        "create_experiment_result",
        Mock(
            return_value=object()
        ),
    )

    monkeypatch.setattr(
        main_module,
        "save_experiment_result",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "visualize_results",
        visualize_results,
    )

    monkeypatch.setattr(
        main_module,
        "_print_latent_statistics",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "save_final_checkpoint",
        Mock(),
    )

    monkeypatch.setattr(
        main_module,
        "GifWriter",
        writer_class,
    )

    monkeypatch.setattr(
        main_module,
        "show_figures",
        show_figures,
    )

    monkeypatch.setattr(
        main_module,
        "close_figures",
        close_figures,
    )

    main_module.run_experiment(
        expected_settings
    )

    writer_class.assert_called_once_with(
        main_module.GifWriterSettings(
            fps=2.0,
            output_width=1920,
            output_height=1080,
            render_dpi=120.0,
        )
    )

    writer.save.assert_called_once_with(
        validation_figures,
        Path(
            "results/validation_latent_space.gif"
        ),
    )

    assert (
        close_figures.call_count
        == 2
    )

    gif_close_call = (
        close_figures.call_args_list[0]
    )

    assert (
        gif_close_call.args[0]
        == validation_figures[2:-2]
    )

    show_figures.assert_called_once_with(
        [training_figure]
    )

    evaluate_validation_set.assert_called_once_with(
        model=expected_model,
        validation_loader=expected_validation_loader,
        settings=expected_settings,
    )

    evaluate_test_set = main_module.evaluate_test_set

    assert not evaluate_test_set.called

    assert close_figures.call_args_list[0].args[0] == (
        validation_figures[2:-2]
    )
