import pathlib

import numpy as np
import PIL
import tensorflow as tf


def load_image(filepath: str | pathlib.Path, height: int, width: int) -> tf.Tensor:
    """Return the resized image from `filename`.

    Parameters
    ----------
    filepath
        Path to file.
    height
        Image height after resizing.
    width
        Image width after resizing.

    Returns
    -------
    image
        Tensorflow tensor of shape (1, width, height, 3)
    """
    image = np.array(PIL.Image.open(filepath).resize((width, height)))  # type: ignore [assignment]
    image = tf.constant(np.reshape(image, (1, *image.shape)))  # type: ignore [assignment]

    return image  # type: ignore [return-value]


def _clip_0_1(image: tf.Tensor) -> tf.Tensor:
    """Clip pixels in the tensor to be between 0 and 1."""
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor: tf.Tensor, keepdims=False) -> PIL.Image:
    """Convert the given tensor into a PIL image.

    Parameters
    ----------
    tensor
        Tensor of shape.
    keepdims
        If `True`, remove the first dimension from the return.

    Returns
    -------
    PIL.Image
        An image object.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)  # type: ignore [assignment]
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        if not keepdims:
            tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
