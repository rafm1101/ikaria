import tensorflow as tf


def get_layer_outputs(model: tf.keras.Model, layer_names: list[str]) -> list[tf.keras.Layer]:
    """Retrieve a list of layer outputs.

    Parameters
    ----------
    model
        Pre-trained tensorflow model.
    layer_names
        List of layer names, supposed to contained in the model.

    Returns
    -------
    layer_outputs
        List of tensors.
    """
    if len(layer_names[0]) == 2:
        layer_names = [x[0] for x in layer_names]

    layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]

    assert len(layer_names) == len(
        layer_outputs
    ), "Did not find all layers in the model. Compare layer names and model's summary."

    return layer_outputs


def create_layer_output_model(model, outputs):
    """Create a model that returns a list of intermediate output values.

    Arguments:
      model -- a tensorflow model, supposed to be the pre-trained one
      outputs -- a list of tensors representing the desired hidden layers to watch
    Returns:
       model with the desired outputs
    """

    return tf.keras.Model(inputs=[model.input], outputs=outputs)
