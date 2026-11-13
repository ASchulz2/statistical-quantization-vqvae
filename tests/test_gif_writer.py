from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from classes.gif_writer import (
    GifWriter,
    GifWriterSettings,
)


def test_gif_writer_saves_all_figures(
    tmp_path: Path,
) -> None:
    figures = []

    for index in range(3):
        figure = plt.figure()

        axis = figure.add_subplot(111)

        axis.scatter(
            [0.0, 1.0],
            [float(index), float(index + 1)],
        )

        axis.set_title(
            f"Epoch {index}"
        )

        figures.append(figure)

    output_path = (
        tmp_path
        / "latent_space.gif"
    )

    writer = GifWriter(
        GifWriterSettings(
            fps=2.0,
        )
    )

    result = writer.save(
        figures,
        output_path,
    )

    assert result == output_path
    assert output_path.exists()

    with Image.open(output_path) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3

    for figure in figures:
        plt.close(figure)


def test_gif_writer_rejects_empty_figures(
    tmp_path: Path,
) -> None:
    writer = GifWriter()

    with pytest.raises(
        ValueError,
        match="At least one figure",
    ):
        writer.save(
            [],
            tmp_path / "empty.gif",
        )


def test_gif_writer_settings_validate() -> None:
    with pytest.raises(
        ValueError,
        match="fps",
    ):
        GifWriterSettings(
            fps=0.0,
        )

    with pytest.raises(
        ValueError,
        match="loop",
    ):
        GifWriterSettings(
            loop=-1,
        )

