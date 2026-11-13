"""
Tests for image preparation utilities.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from classes.image_preparation import (
    flattened_to_image,
    prepare_images,
)


def test_mnist_flattened_batch() -> None:
    images = np.zeros(
        (4, 784),
        dtype=np.float32,
    )

    result = prepare_images(
        images
    )

    assert result.shape == (
        4,
        28,
        28,
    )


def test_cifar10_flattened_batch() -> None:
    images = np.zeros(
        (4, 3072),
        dtype=np.float32,
    )

    result = prepare_images(
        images
    )

    assert result.shape == (
        4,
        32,
        32,
        3,
    )


def test_nchw_batch_is_converted_to_nhwc() -> None:
    images = np.zeros(
        (4, 3, 32, 32),
        dtype=np.float32,
    )

    result = prepare_images(
        images
    )

    assert result.shape == (
        4,
        32,
        32,
        3,
    )


def test_nchw_grayscale_batch_loses_channel() -> None:
    images = np.zeros(
        (4, 1, 28, 28),
        dtype=np.float32,
    )

    result = prepare_images(
        images
    )

    assert result.shape == (
        4,
        28,
        28,
    )


def test_nhwc_batch_is_preserved() -> None:
    images = np.zeros(
        (4, 32, 32, 3),
        dtype=np.float32,
    )

    result = prepare_images(
        images
    )

    assert result.shape == (
        4,
        32,
        32,
        3,
    )


def test_single_flattened_image() -> None:
    image = np.zeros(
        784,
        dtype=np.float32,
    )

    result = prepare_images(
        image
    )

    assert result.shape == (
        1,
        28,
        28,
    )


def test_single_chw_image() -> None:
    image = np.zeros(
        (3, 32, 32),
        dtype=np.float32,
    )

    result = prepare_images(
        image
    )

    assert result.shape == (
        32,
        32,
        3,
    )


def test_single_grayscale_image() -> None:
    image = np.zeros(
        (28, 28),
        dtype=np.float32,
    )

    result = prepare_images(
        image
    )

    assert result.shape == (
        1,
        28,
        28,
    )


def test_single_hwc_grayscale_image() -> None:
    image = np.zeros(
        (28, 28, 1),
        dtype=np.float32,
    )

    result = prepare_images(
        image
    )

    assert result.shape == (
        28,
        28,
    )


def test_torch_tensor_is_detached_and_converted() -> None:
    images = torch.full(
        (2, 784),
        2.0,
        requires_grad=True,
    )

    result = prepare_images(
        images
    )

    assert isinstance(
        result,
        np.ndarray,
    )

    assert result.shape == (
        2,
        28,
        28,
    )

    assert np.all(
        result == 1.0
    )


def test_values_are_clipped() -> None:
    images = np.array(
        [
            -1.0,
            0.5,
            2.0,
        ]
    ).reshape(
        1,
        3,
    )

    #
    # Use a supported flattened dimension by constructing
    # an MNIST-sized array.
    #

    mnist = np.zeros(
        (1, 784),
        dtype=np.float32,
    )

    mnist[0, 0] = -1.0
    mnist[0, 1] = 0.5
    mnist[0, 2] = 2.0

    result = prepare_images(
        mnist
    )

    assert result.min() >= 0.0
    assert result.max() <= 1.0

    assert result[0, 0, 0] == 0.0
    assert result[0, 0, 1] == 0.5
    assert result[0, 0, 2] == 1.0


@pytest.mark.parametrize(
    "shape",
    [
        (1, 100),
        (4, 100),
        (2, 2, 2, 2),
        (2, 2, 2),
    ],
)
def test_unsupported_shapes_raise(
    shape: tuple[int, ...],
) -> None:
    images = np.zeros(
        shape,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        prepare_images(
            images
        )


def test_flattened_to_image_requires_two_dimensions() -> None:
    images = np.zeros(
        784,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        flattened_to_image(
            images
        )


def test_flattened_to_image_rejects_unknown_dimension() -> None:
    images = np.zeros(
        (4, 100),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        flattened_to_image(
            images
        )