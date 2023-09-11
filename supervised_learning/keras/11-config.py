#!/usr/bin/env python3
"""Save and Load Configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model’s configuration in JSON format.

    Args:
        network (Model): the model whose configuration should be saved.
        filename (str):  the path of the file that the configuration should be
        saved to.
    """

    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)


def load_config(filename):
    """Loads a model with a specific configuration.

    Args:
        filename (str): the path of the file containing the model’s
        configuration in JSON format.

    Returns:
        Model: the loaded model.
    """

    with open(filename, 'r') as f:
        model_config = f.read()
    return K.models.model_from_json(model_config)
