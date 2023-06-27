# Copyrights (c) Preetham Ganesh.


import os
import sys


BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)


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

    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """Input tensor is passed through the layers in the model.

        Input tensor is passed through the layers in the model.

        Args:
            inputs: A list of tensors containing inputs for
        """
        # Asserts type & values of the input arguments.
        assert isinstance(inputs, list), "Variable inputs should be of type 'list'."
        assert isinstance(training, bool), "Variable training should be of type 'bool'."

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


class CifarClassificationCNN(tf.Module):
    """Recognizes object in an image."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes components in the module.

        Initializes components in the module.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration

    def load_model(self) -> None:
        """Loads model & other utilies for training it.

        Loads model & other utilies for training it.

        Args:
            None.

        Returns:
            None.
        """
        # Loads model for current model configuration.
        self.model = Model(self.model_configuration)

        # Based on the name & configuration, optimizer is initialized.
        if (
            self.model_configuration["cifar_classfication"]["optimizer"]["name"]
            == "adam"
        ):
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.model_configuration["cifar_classfication"][
                    "optimizer"
                ]["learning_rate"]
            )

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        self.home_directory_path = os.getcwd()
        self.checkpoint_directory_path = (
            "{}/models/cifar_classfication/v{}/checkpoints".format(
                self.home_directory_path, self.model_configuration["version"]
            )
        )
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.model
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_directory_path, max_to_keep=3
        )
