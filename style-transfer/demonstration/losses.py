import tensorflow as tf

DEFAULT_STYLE_LAYERS = [
    ("block1_conv1", 0.2),
    ("block2_conv1", 0.2),
    ("block3_conv1", 0.2),
    ("block4_conv1", 0.2),
    ("block5_conv1", 0.2),
]


@tf.function
def compute_total_cost(J_content: tf.Tensor, J_style: tf.Tensor, alpha: float = 10, beta: float = 40) -> tf.Tensor:
    """Compute a weighted total cost of two losses.

    Parameters
    ----------
    J_content
        Loss value of content loss.
    J_style
        Loss value of style loss.
    alpha
        Weight of content loss. Hyperparameter.
    beta
        Weight of style loss. Hyperparameter.

    Returns
    -------
    J
        Total loss.
    """
    J = alpha * J_content + beta * J_style
    return J


@tf.function()
def compute_content_cost(content_output: tf.Tensor, generated_output: tf.Tensor) -> tf.Tensor:
    """Compute the content cost.

    Parameters
    ----------
    content_output
        Tensor of shape (1, height, width, n_channels).
    generated_output
        Tensor of shape (1, height, width, n_channels).

    Returns
    -------
    J_content
        Loss due to content. Tensor of shape (1,).

    Notes
    -----
      a_C --  intermediate representation of content of image C
              tensor of shape (1, n_H, n_W, n_C)
      a_G --  intermediate representation of content of image C
              tensor of shape (1, n_H, n_W, n_C)

    """
    a_C = content_output[-1]
    a_G = generated_output[-1]

    m, height, width, n_channels = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[m, -1, n_channels]), perm=[0, 2, 1])
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[m, -1, n_channels]), perm=[0, 2, 1])

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (
        4 * height * width * n_channels  # type: ignore [operator]
    )

    return J_content


def compute_layer_style_cost(a_S: tf.Tensor, a_G: tf.Tensor) -> tf.Tensor:
    """little helper function
    Compute style cost of a given layer

    Parameters
    ----------
      a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
      a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns
    -------
    J_style_layer
        Loss due to style. Tensor of shape (1,).

    Notes
    -----
    1. Tensors are reshaped to (n_channels, height*width).
    """
    m, height, width, n_channels = a_G.get_shape().as_list()

    a_S = tf.squeeze(
        tf.transpose(
            tf.reshape(a_S, shape=[-1, height * width, n_channels]), perm=[2, 1, 0]  # type: ignore [operator]
        ),
        axis=2,
    )
    a_G = tf.squeeze(
        tf.transpose(
            tf.reshape(a_G, shape=[-1, height * width, n_channels]), perm=[2, 1, 0]  # type: ignore [operator]
        ),
        axis=2,
    )

    gram_S = _compute_gram_matrix(a_S)
    gram_G = _compute_gram_matrix(a_G)

    J_style_layer = (
        tf.reduce_sum(tf.square(tf.subtract(gram_S, gram_G)))
        / (2 * height * width * n_channels) ** 2  # type: ignore [operator]
    )

    return J_style_layer


def _compute_gram_matrix(A: tf.Tensor) -> tf.Tensor:
    """Compute the Gram matrix of a tensor.

    Parameters
    ----------
    A
        Tensor of shape (n_channels, height*width).

    Returns
    -------
    GA
        Tensor of shape (n_channels, n_channels).
    """
    GA = tf.matmul(A, tf.transpose(A))
    return GA


@tf.function
def compute_style_cost(style_image_output, generated_image_output, style_layers=DEFAULT_STYLE_LAYERS):
    """Compute the overall style cost from given layers.

    Parameters
    ----------
    style_image_output
        Tensorflow model's hidden layers representations.
    generated_image_output
        Tensorflow model's hidden layers representations.
    style_layers
        A list containing:
        - The names of the layers we would like to extract style from.
        - A coefficient for each of them.

    Returns
    -------
    J_style
        Style loss value. Tensor of shape (1,).

    Notes
    -----
    1. The observed layers are the style layers and the content layer (whose feature map is removed first).
    """
    J_style = 0

    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]

    for i, weight in zip(range(len(a_S)), style_layers):
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer

    return J_style
