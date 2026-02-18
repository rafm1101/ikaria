import images
import losses
import tensorflow as tf


@tf.function()
def training_step(
    generated_image: tf.Tensor, a_S: tf.Tensor, a_C: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.Optimizer
):
    """Perform a single training step for generating the style transferred image.

    Parameters
    ----------
    generated_image
        The image to be optimised.
    a_S
        Tensor of shape (), model outputs of the style image.
    a_C
        Tensor of shape (), model outputs of the content image.

    Returns
    -------
    J
        Loss.
    """
    with tf.GradientTape() as tape:
        tape.watch(generated_image)

        a_G = model(generated_image)

        J_style = losses.compute_style_cost(a_S[:-1], a_G[:-1])
        J_content = losses.compute_content_cost(a_C, a_G)
        J = losses.compute_total_cost(J_content, J_style)

    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(images._clip_0_1(generated_image))

    return J
