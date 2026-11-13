from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image


@dataclass(frozen=True)
class GifWriterSettings:
    """
    Configuration for GIF generation from Matplotlib figures.
    """

    fps: float = 2.0
    loop: int = 0

    output_width: int = 1920
    output_height: int = 1080
    render_dpi: float = 120.0

    def __post_init__(self) -> None:
        if self.fps <= 0.0:
            raise ValueError(
                "fps must be positive."
            )

        if self.loop < 0:
            raise ValueError(
                "loop must be non-negative."
            )

        if self.output_width <= 0:
            raise ValueError(
                "output_width must be positive."
            )

        if self.output_height <= 0:
            raise ValueError(
                "output_height must be positive."
            )

        if self.render_dpi <= 0.0:
            raise ValueError(
                "render_dpi must be positive."
            )


class GifWriter:
    """
    Convert an ordered sequence of Matplotlib figures into a GIF.
    """

    def __init__(
        self,
        settings: GifWriterSettings | None = None,
    ) -> None:
        if settings is None:
            settings = GifWriterSettings()

        self.settings = settings

    def _figure_to_image(
        self,
        figure: Figure,
    ) -> Image.Image:
        """
        Render one Matplotlib figure at the configured output resolution.
        """
        original_size = figure.get_size_inches().copy()
        original_dpi = figure.get_dpi()

        width_inches = (
            self.settings.output_width
            /
            self.settings.render_dpi
        )

        height_inches = (
            self.settings.output_height
            /
            self.settings.render_dpi
        )

        try:
            figure.set_size_inches(
                width_inches,
                height_inches,
                forward=True,
            )

            figure.set_dpi(
                self.settings.render_dpi
            )

            figure.canvas.draw()

            rgba = np.asarray(
                figure.canvas.buffer_rgba()
            )

            return Image.fromarray(
                rgba[..., :3].copy(),
                mode="RGB",
            )

        finally:
            figure.set_size_inches(
                original_size,
                forward=True,
            )

            figure.set_dpi(
                original_dpi
            )

    def save(
        self,
        figures,
        output_path: str | Path,
    ) -> Path:
        """
        Save an ordered sequence of figures as a GIF.
        """
        figures = list(figures)

        if not figures:
            raise ValueError(
                "At least one figure is required."
            )

        output_path = Path(
            output_path
        )

        images = [
            self._figure_to_image(
                figure
            )
            for figure in figures
        ]

        first_size = images[0].size

        if any(
            image.size != first_size
            for image in images[1:]
        ):
            raise ValueError(
                "All figures must render to the same image size."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        duration = max(
            1,
            round(
                1000.0
                / self.settings.fps
            ),
        )

        images[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=self.settings.loop,
        )

        return output_path