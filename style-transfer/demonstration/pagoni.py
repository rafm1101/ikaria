import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


def plot_row(images, titles, n_images=3, n=3, J=np.zeros((1, 0))):
    """
    plot images next to each other with titles

    Parameters
    ----------
    images
        Tuple or tensor of images.
    titles
        Tuple or tensor of titles.
    n_images
        Number of images to display.
    row_max
        Max number of images per row.
      J - array of values of a function to display together with the images
          (typically the loss function at the end of the loops through the epochs)
    """
    fig = plt.figure(figsize=(16, 4))

    for i in range(np.minimum(n_images, images.shape[0])):
        ax = fig.add_subplot(1, n, i + 1)
        if not tf.is_tensor(images):
            plt.imshow(images[i])
        else:
            plt.imshow(images[i].numpy())
        ax.title.set_text(titles[i])

    if J.size:
        if n_images == 3:
            plt.show()
            fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(1, n, n_images % 3 + 1)
        plt.plot(J)
        ax.title.set_text("style loss")

    plt.show()


def show_layer_representations(feature_maps, layer_names: list[str]):
    """Display feature maps

    Parameters
    ----------
    feature_maps
        Predictions of the model's hidden layers.
    layer_names
        Names of the layers (list of str)

    Notes
    -----
    1. The shape of the feature map is expected to be (1, h, w, n).
    2. Shows the features of the convolution and pooling layers only.
    """
    for layer_name, feature_map in zip(layer_names, feature_maps):

        if len(feature_map.shape) == 4:
            n_channels = feature_map.shape[-1]
            height, width = feature_map.shape[1:3]

            display_grid = np.zeros((height, width * n_channels))
            for i in range(n_channels):
                display_grid[:, i * width : (i + 1) * width] = _feature_map_to_image(feature_map[0, :, :, i])

            scale = 200.0 / n_channels

            plt.figure(figsize=(scale * n_channels, scale))
            plt.title(layer_name)
            plt.imshow(display_grid, aspect="auto", cmap="magma")


def _feature_map_to_image(X: NDArray[np.floating]):
    """Standardize and scale a feature map.

    Parameters
    ----------
    X
      Slice of a feature map, of shape ( h, w ).

    Returns
    -------
    np.array
        Clipped np.ndarray ( h, w ) dtype uint8.
    """
    X = X.copy()
    s = X.std()
    X = (X - X.mean()) / s if s > 0 else X - X.mean()
    X = X * 64 + 128

    return np.clip(X, 0, 255).astype("uint8")
