# Copyrights (c) Preetham Ganesh.


import tensorflow as tf
from typing import Dict, Any, List


class Model(tf.keras.Model):
    """A tensorflow model to recognize object in an image."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes the layers in the recognition model.

        Initializes the layers in the recognition model, by adding convolutional, pooling, dropout & dense layers.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        super(Model, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration
        self.model_layers = dict()

        # Iterates across layers in the layers arrangement.
        self.model_layers = dict()
        for name in self.model_configuration["cifar_classification"]["arrangement"]:
            layer = self.model_configuration["cifar_classification"]["configuration"][
                name
            ]

            # If layer's name is like 'conv2d_', a Conv2D layer is initialized based on layer configuration.
            if name.split("_")[0] == "conv2d":
                self.model_layers[name] = tf.keras.layers.Conv2D(
                    filters=layer["filters"],
                    kernel_size=layer["kernel_size"],
                    padding=layer["padding"],
                    strides=layer["strides"],
                    activation=layer["activation"],
                    name=name,
                )

            # If layer's name is like 'maxpool2d_', a MaxPool2D layer is initialized based on layer configuration.
            elif name.split("_")[0] == "maxpool2d":
                self.model_layers[name] = tf.keras.layers.MaxPool2D(
                    pool_size=layer["pool_size"],
                    strides=layer["strides"],
                    padding=layer["padding"],
                    name=name,
                )

            # If layer's name is like 'dropout_', a Dropout layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dropout":
                self.model_layers[name] = tf.keras.layers.Dropout(rate=layer["rate"])

            # If layer's name is like 'dense_', a Dropout layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dense":
                self.model_layers[name] = tf.keras.layers.Dense(
                    units=layer["units"],
                    activation=layer["activation"],
                    name=name,
                )

            # If layer's name is like 'flatten_', a Flatten layer is initialized.
            elif name.split("_")[0] == "flatten":
                self.model_layers[name] = tf.keras.layers.Flatten(name=name)

            # If layer's name is like 'resizing_', a Resizing layer is initialized.
            elif name.split("_")[0] == "resizing":
                self.model_layers[name] = tf.keras.layers.Resizing(
                    height=layer["height"],
                    width=layer["width"],
                    interpolation=layer["interpolation"],
                )

    def call(
        self,
        inputs: List[tf.Tensor],
        training: bool = False,
        masks: List[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Input tensor is passed through the layers in the model.

        Input tensor is passed through the layers in the model.

        Args:
            inputs: A list of tensors containing inputs for the model's prediction.
            training: A boolean value for the flag of training/testing state.
            masks: A list of tensors containing masks for the model's prediction.

        Returns:
            A tensor for the prediction from the model for the current inputs & masks.
        """
        # Asserts type & values of the input arguments.
        assert isinstance(inputs, list), "Variable inputs should be of type 'list'."
        assert isinstance(training, bool), "Variable training should be of type 'bool'."
        assert (
            isinstance(masks, list) or masks is None
        ), "Variable masks should be of type 'list' or masks should have value as 'None'."

        # Iterates across the layers arrangement, and predicts the output for each layer.
        x = inputs[0]
        for name in self.model_configuration["cifar_classification"]["arrangement"]:
            # If layer's name is like 'dropout_', the following output is predicted.
            if name.split("_")[0] == "dropout":
                x = self.model_layers[name](x, training=training)

            # Else, the following output is predicted.
            else:
                x = self.model_layers[name](x)
        return x

    def build_graph(self) -> tf.keras.Model:
        """Builds plottable graph for the model."""
        # Creates the input layer using the model configuration.
        inputs = [
            tf.keras.layers.Input(
                shape=(
                    self.model_configuration["final_image_height"],
                    self.model_configuration["final_image_width"],
                    self.model_configuration["n_channels"],
                )
            )
        ]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
