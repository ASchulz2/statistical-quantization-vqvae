"""
Image preparation utilities for visualization.

This module contains pure functions for converting tensors and
NumPy arrays into layouts accepted by Matplotlib.

Supported image representations include:

    (N, D)
    (N, C, H, W)
    (N, H, W, C)
    (D,)
    (C, H, W)
    (H, W)
    (H, W, C)

Flattened MNIST and CIFAR-10 representations are also supported:

    784  -> 28 x 28
    3072 -> 32 x 32 x 3
"""

from __future__ import annotations

import numpy as np
import torch


def flattened_to_image(
    images: np.ndarray,
) -> np.ndarray:
    """
    Convert flattened image batches to image layouts.

    Parameters
    ----------
    images:
        Two-dimensional array with shape ``(N, D)``.

    Returns
    -------
    numpy.ndarray
        Image batch with either shape ``(N, 28, 28)`` for MNIST
        or ``(N, 32, 32, 3)`` for CIFAR-10.

    Raises
    ------
    ValueError
        If the input does not have shape ``(N, D)`` or if the
        flattened dimension is unsupported.
    """
    if images.ndim != 2:
        raise ValueError(
            "Expected flattened images with shape "
            f"(N, D). Got {images.shape}."
        )

    dimension = images.shape[1]

    if dimension == 28 * 28:
        return images.reshape(
            -1,
            28,
            28,
        )

    if dimension == 32 * 32 * 3:
        return (
            images.reshape(
                -1,
                3,
                32,
                32,
            )
            .transpose(
                0,
                2,
                3,
                1,
            )
        )

    raise ValueError(
        "Unsupported flattened image dimension: "
        f"{dimension}."
    )


def prepare_images(
    images: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """
    Convert supported image representations into Matplotlib layout.

    Parameters
    ----------
    images:
        Image tensor or NumPy array.

        Supported shapes are:

            ``(N, D)``
                flattened image batch

            ``(N, C, H, W)``
                batch in PyTorch NCHW layout

            ``(N, H, W, C)``
                batch in Matplotlib NHWC layout

            ``(D,)``
                single flattened image

            ``(C, H, W)``
                single PyTorch image

            ``(H, W)``
                single grayscale image

            ``(H, W, C)``
                single Matplotlib image

    Returns
    -------
    numpy.ndarray
        Matplotlib-ready image array.

        Grayscale channels are reduced from ``(..., 1)`` to ``(...)``.
        Pixel values are clipped to ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input shape is unsupported.
    """
    if isinstance(
        images,
        torch.Tensor,
    ):
        images = (
            images.detach()
            .cpu()
            .numpy()
        )

    images = np.asarray(
        images
    )

    if images.ndim == 0:
        raise ValueError(
            "Unsupported scalar image input."
        )

    if images.ndim == 1:
        images = flattened_to_image(images.reshape(1, -1))

    elif images.ndim == 2:
        if (
            images.shape[0] in {28, 32}
            and images.shape[1] in {28, 32}
        ):
            images = images[np.newaxis, ...]
        else:
            images = flattened_to_image(images)

    #
    # Image batch.
    #

    elif images.ndim == 4:

        #
        # NCHW -> NHWC.
        #

        if images.shape[1] in (1, 3):

            images = images.transpose(
                0,
                2,
                3,
                1,
            )

        #
        # NHWC.
        #

        elif images.shape[-1] in (1, 3):
            pass

        else:
            raise ValueError(
                "Unsupported 4D image shape: "
                f"{images.shape}."
            )

    #
    # Single image.
    #

    elif images.ndim == 3:

        #
        # CHW -> HWC.
        #
        # The last dimension is not a channel dimension.
        #

        if (
            images.shape[0] in (1, 3)
            and
            images.shape[-1] not in (1, 3)
        ):
            images = images.transpose(
                1,
                2,
                0,
            )

        #
        # HWC.
        #

        elif images.shape[-1] in (1, 3):
            pass

        #
        # Unsupported 3D layout.
        #

        else:
            raise ValueError(
                "Unsupported 3D image shape: "
                f"{images.shape}."
            )

    else:
        raise ValueError(
            "Unsupported image shape: "
            f"{images.shape}."
        )

    #
    # Remove singleton grayscale channel.
    #

    if (
        images.ndim == 4
        and
        images.shape[-1] == 1
    ):
        images = images[..., 0]

    elif (
        images.ndim == 3
        and
        images.shape[-1] == 1
    ):
        images = images[..., 0]

    return np.clip(
        images,
        0.0,
        1.0,
    )