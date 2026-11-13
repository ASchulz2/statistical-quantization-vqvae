from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import csv


# ============================================================
# Experiment result
# ============================================================

@dataclass
class ExperimentResult:
    """
    Canonical result record for one completed experiment.

    The same schema is used for:

        * classical VQ
        * predictive VQ
        * VQ-VAE
        * statistical VQ-VAE
        * rate-distortion experiments

    Configuration fields describe how the experiment was run.
    Metric fields describe the resulting model.
    """

    # --------------------------------------------------------
    # Experiment identity
    # --------------------------------------------------------

    dataset: str

    method: str

    predictive: bool = False

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    latent_dim: int | None = None

    num_codewords: int | None = None

    epochs: int | None = None

    batch_size: int | None = None

    learning_rate: float | None = None

    commitment_beta: float | None = None

    use_ema: bool | None = None

    use_variance: bool | None = None

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed: int | None = None

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    runtime: float | None = None

    # --------------------------------------------------------
    # Reconstruction metrics
    # --------------------------------------------------------

    distortion: float | None = None

    distortion_db: float | None = None

    rmse: float | None = None

    psnr: float | None = None

    # --------------------------------------------------------
    # Discrete representation metrics
    # --------------------------------------------------------

    rate: float | None = None

    perplexity: float | None = None

    utilization: float | None = None

    entropy: float | None = None

    active_codewords: int | None = None

    dead_codewords: int | None = None

    # --------------------------------------------------------
    # Validation information
    # --------------------------------------------------------

    best_val_loss: float | None = None

    best_val_distortion: float | None = None

    best_val_loss_epoch: int | None = None

    best_val_distortion_epoch: int | None = None

    # --------------------------------------------------------
    # Additional information
    # --------------------------------------------------------

    extra: dict[str, Any] | None = None

    # ========================================================
    # Conversion
    # ========================================================

    def to_row(
        self,
    ) -> dict[str, Any]:
        """
        Convert the result into a flat CSV-compatible row.
        """

        row = asdict(
            self
        )

        extra = row.pop(
            "extra",
            None,
        )

        if extra is not None:

            for key, value in extra.items():

                if key not in row:

                    row[key] = value

        return row


# ============================================================
# Settings
# ============================================================

@dataclass
class ExperimentResultsSettings:
    """
    Configuration for result storage.
    """

    output_path: str = (
        "results/experiments.csv"
    )

    overwrite: bool = True

    include_header: bool = True


# ============================================================
# Results collection
# ============================================================

class ExperimentResults:
    """
    Collection and CSV exporter for experiment results.
    """

    CORE_FIELDS = [

        # Identity
        "dataset",
        "method",
        "predictive",

        # Configuration
        "latent_dim",
        "num_codewords",
        "epochs",
        "batch_size",
        "learning_rate",
        "commitment_beta",
        "use_ema",
        "use_variance",

        # Reproducibility
        "seed",

        # Runtime
        "runtime",

        # Reconstruction
        "distortion",
        "distortion_db",
        "rmse",
        "psnr",

        # Discrete representation
        "rate",
        "perplexity",
        "utilization",
        "entropy",
        "active_codewords",
        "dead_codewords",

        # Validation
        "best_val_loss",
        "best_val_distortion",
        "best_val_loss_epoch",
        "best_val_distortion_epoch",

    ]


    def __init__(
        self,
        settings: ExperimentResultsSettings | None = None,
    ) -> None:

        self.settings = (
            settings
            if settings is not None
            else ExperimentResultsSettings()
        )

        self.results: list[
            ExperimentResult
        ] = []


    # ========================================================
    # Add
    # ========================================================

    def add(
        self,
        result: ExperimentResult,
    ) -> None:

        self.results.append(
            result
        )


    # ========================================================
    # Add from dictionary
    # ========================================================

    def add_dict(
        self,
        values: dict[str, Any],
    ) -> None:
        """
        Construct and add an ExperimentResult from a dictionary.

        This is useful while migrating existing experiments whose
        output is already represented as dictionaries.
        """

        known = set(
            self.CORE_FIELDS
        )


        result_values = {

            key: values[key]

            for key in known

            if key in values

        }


        extra = {

            key: value

            for key, value in values.items()

            if key not in known

        }


        if "dataset" not in result_values:

            result_values["dataset"] = (
                "unknown"
            )


        if "method" not in result_values:

            raise ValueError(
                "Experiment result requires "
                "'method'."
            )


        result_values["extra"] = (
            extra
            if extra
            else None
        )


        result = ExperimentResult(
            **result_values
        )


        self.add(
            result
        )


    # ========================================================
    # Extend
    # ========================================================

    def extend(
        self,
        results: Iterable[
            ExperimentResult
        ],
    ) -> None:

        for result in results:

            self.add(
                result
            )


    # ========================================================
    # Clear
    # ========================================================

    def clear(
        self,
    ) -> None:

        self.results.clear()


    # ========================================================
    # Rows
    # ========================================================

    def rows(
        self,
    ) -> list[dict[str, Any]]:

        return [

            result.to_row()

            for result in self.results

        ]


    # ========================================================
    # Field names
    # ========================================================

    def fieldnames(
        self,
    ) -> list[str]:
        """
        Return the canonical fields followed by any
        additional fields discovered through ``extra``.
        """

        fields = list(
            self.CORE_FIELDS
        )


        for row in self.rows():

            for key in row.keys():

                if key not in fields:

                    fields.append(
                        key
                    )


        return fields


    # ========================================================
    # Save CSV
    # ========================================================

    def save_csv(
        self,
        path: str | Path | None = None,
    ) -> Path:

        output_path = Path(

            path
            if path is not None
            else self.settings.output_path

        )


        if not self.results:

            raise ValueError(
                "Cannot save an empty experiment "
                "result collection."
            )


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        mode = (

            "w"
            if self.settings.overwrite

            else

            "x"

        )


        rows = self.rows()


        with output_path.open(

            mode,

            newline="",

            encoding="utf-8",

        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=self.fieldnames(),

                extrasaction="ignore",

            )


            if self.settings.include_header:

                writer.writeheader()


            for row in rows:

                writer.writerow(
                    row
                )


        return output_path


    # ========================================================
    # Complete LaTeX table
    # ========================================================

    def save_latex_table(
        self,
        path: str | Path,
        columns: list[tuple[str, str]],
        methods: list[str] | None = None,
        predictive: bool | None = None,
        precision: int = 5,
        caption: str | None = None,
        label: str | None = None,
    ) -> Path:
        """
        Write a complete LaTeX tabular environment.

        Parameters
        ----------
        path:
            Output .tex path.

        columns:
            List of (field_name, latex_header) pairs.

        methods:
            Optional method filter.

        predictive:
            Optional predictive/non-predictive filter.

        precision:
            Number of decimal places for distortion.

        caption, label:
            Accepted for convenience but deliberately not written into
            the tabular itself. The surrounding table environment in
            the article should own the caption and label.
        """

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected = self.results

        if methods is not None:

            selected = [
                result
                for result in selected
                if result.method in methods
            ]

        if predictive is not None:

            selected = [
                result
                for result in selected
                if result.predictive == predictive
            ]

        if not selected:

            raise ValueError(
                "No experiment results match the "
                "requested LaTeX table filter."
            )

        alignment = "l" + "r" * (
            len(columns) - 1
        )

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:

            file.write(
                f"\\begin{{tabular}}{{{alignment}}}\n"
            )

            file.write(
                "\\toprule\n"
            )

            headers = [
                header
                for _, header in columns
            ]

            file.write(
                " & ".join(headers)
                + " \\\\\n"
            )

            file.write(
                "\\midrule\n"
            )

            for result in selected:

                values = []

                for field, _ in columns:

                    value = getattr(
                        result,
                        field,
                    )

                    if value is None:

                        raise ValueError(
                            f"Missing field "
                            f"'{field}' for "
                            f"{result.method}."
                        )

                    #
                    # Format numeric values.
                    #

                    if field == "distortion":

                        formatted = (
                            f"{float(value):.{precision}f}"
                        )

                    elif field in {
                        "rate",
                        "perplexity",
                        "utilization",
                    }:

                        formatted = (
                            f"{float(value):.3f}"
                        )

                    else:

                        formatted = str(
                            value
                        )

                    values.append(
                        formatted
                    )

                #
                # Escape method names if necessary.
                #

                if values:

                    if result.method == "KMeans":

                        values[0] = (
                            r"K--Means"
                        )

                    elif result.method == "Competitive":

                        values[0] = (
                            r"Competitive learning"
                        )

                file.write(
                    " & ".join(values)
                    + " \\\\\n"
                )

            file.write(
                "\\bottomrule\n"
            )

            file.write(
                "\\end{tabular}\n"
            )

        return output_path

    def append_csv(
        self,
        path: str | Path | None = None,
    ) -> Path:
        """
        Append collected results to an existing CSV.

        Creates the file and writes the header when it does not
        yet exist.
        """

        if path is None:
            path = self.settings.output_path

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = self.rows()

        if not rows:
            raise ValueError(
                "Cannot append an empty result collection."
            )

        file_exists = output_path.exists()

        with output_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=self.fieldnames(),

                extrasaction="ignore",

            )

            if (
                not file_exists
                and self.settings.include_header
            ):
                writer.writeheader()

            for row in rows:
                writer.writerow(row)

        return output_path

    

# ============================================================
# Convenience function
# ============================================================

def save_results_csv(
    results: Iterable[
        ExperimentResult
    ],
    path: str | Path = (
        "results/experiments.csv"
    ),
) -> Path:

    collection = ExperimentResults(

        ExperimentResultsSettings(
            output_path=str(path)
        )

    )


    collection.extend(
        results
    )


    return collection.save_csv(
        path
    )