# Copyrights (c) Preetham Ganesh.


import os
import sys
import warnings
import argparse
import logging


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


import tensorflow as tf
import pandas as pd

from src.utils import load_json_file
from src.utils import create_log
from src.utils import add_to_log
from src.utils import set_physical_devices_memory_limit
from src.cifar_classification.model import Model
from src.cifar_classification.dataset import Dataset
from src.utils import check_directory_path_existence
from src.utils import save_json_file


class Train(object):
    """Trains the Cifar classification CNN model based on model configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the model to be trained.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.best_validation_loss = None

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for current version.

        Loads the model configuration file for current version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = (
            "{}/configs/models/cifar_classification".format(self.home_directory_path)
        )
        self.model_configuration = load_json_file(
            "v{}".format(self.model_version), model_configuration_directory_path
        )

    def load_dataset(self) -> None:
        """Loads dataset based on dataset version in the model configuration.

        Loads dataset based on dataset version in the model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Creates object attributes for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Loads the training, validation, and testing labels created when downloading the dataset.
        self.dataset.load_data()

        # Creates a dictionary to store the unique label ids & names.
        self.model_configuration["labels"] = self.dataset.map_label_ids_names()

        # Converts images id & label id into tensorflow dataset and slices them based on batch size.
        self.dataset.shuffle_slice_datasets()

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

    def generate_model_summary_and_plot(self, plot: bool) -> None:
        """Generates summary and plot for loaded model.

        Generates summary and plot for loaded model.

        Args:
            pool: A boolean value to whether generate model plot or not.

        Returns:
            None.
        """
        # Builds plottable graph for the model.
        model = self.model.build_graph()

        # Compiles the model to log the model summary.
        model_summary = list()
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        add_to_log(model_summary)
        add_to_log("")

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            "models/cifar_classification/v{}/reports".format(
                self.model_configuration["version"]
            )
        )

        # Plots the model & saves it as a PNG file.
        if plot:
            model_plot_path = "{}/model.png".format(self.reports_directory_path)
            tf.keras.utils.plot_model(
                model,
                model_plot_path,
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )
            add_to_log("Model plot saved at {}.".format(model_plot_path))
            add_to_log("")

    def initialize_model_history(self) -> None:
        """Creates dicionary for saving the model's history.

        Creates dictionary for saving the model's epoch-wise training and validation metrics history.

        Args:
            None.

        Returns:
            None.
        """
        self.model_history = {
            "epoch": list(),
            "train_loss": list(),
            "validation_loss": list(),
            "train_accuracy": list(),
            "validation_accuracy": list(),
        }

    def initialize_metric_trackers(self) -> None:
        """Initializes TensorFlow trackers which computes the mean of all elements.

        Initializes TensorFlow trackers which computes the mean of all elements.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.validation_accuracy = tf.keras.metrics.Mean(name="validation_accuracy")

    def update_model_history(self, epoch: int) -> None:
        """Updates model history dataframe with latest metrics & saves it as JSON file.

        Updates model history dataframe with latest metrics & saves it as JSON file.

        Args:
            epoch: An integer for the current epoch.

        Returns:
            None.
        """
        self.model_history["epoch"].append(epoch + 1)
        self.model_history["train_loss"].append(
            str(round(self.train_loss.result().numpy(), 3))
        )
        self.model_history["validation_loss"].append(
            str(round(self.validation_loss.result().numpy(), 3))
        )
        self.model_history["train_accuracy"].append(
            str(round(self.train_accuracy.result().numpy(), 3))
        )
        self.model_history["validation_accuracy"].append(
            str(round(self.validation_accuracy.result().numpy(), 3))
        )

        # Saves the model history dictionary as a JSON file.
        save_json_file(self.model_history, "history", self.reports_directory_path)

    def initialize_metrics(self) -> None:
        """Initializes loss & metric function for training the model.

        Initializes loss & metric function for training the model.

        Args:
            None.

        Returns:
            None.
        """
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self.accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()

    def compute_loss(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual values and predicted values.

        Computes loss for the current batch using actual values and predicted values.

        Args:
            target_batch: A tensor for the the actual values for the current batch.
            predicted_batch: A tensor for the predicted values for the current batch.

        Returns:
            A tensor for the loss for the current batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes loss for the current batch using actual values and predicted values.
        loss = self.loss_object(target_batch, predicted_batch)
        return loss

    def compute_accuracy(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes accuracy for the current batch using actual values and predicted values.

        Computes accuracy for the current batch using actual values and predicted values.

        Args:
            target_batch: A tensor which contains the actual values for the current batch.
            predicted_batch: A tensor which contains the predicted values for the current batch.

        Returns:
            A tensor for the accuracy of current batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Resets accuracy object's states.
        self.accuracy_object.reset_state()

        # Computes accuracy for the current batch using actual values and predicted values.
        accuracy = self.accuracy_object(target_batch, predicted_batch)
        return accuracy

    @tf.function
    def train_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Train the current model using current input & target batches.

        Train the current model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for training the model.
            target_batch: A tensor for the target text from the current batch for training and validating the model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes the model output for current batch, and metrics for current model output.
        with tf.GradientTape() as tape:
            predictions = self.model([input_batch], True, None)
            loss = self.compute_loss(target_batch, predictions)
            accuracy = self.compute_accuracy(target_batch, predictions)

        # Computes gradients using loss and model variables.
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Uses optimizer to apply the computed gradients on the combined model variables.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Computes batch metrics and appends it to main metrics.
        self.train_loss(loss)
        self.train_accuracy(accuracy)

    def validation_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Validates the current model using current input & target batches.

        Validates the current model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for validating the model.
            target_batch: A tensor for the target text from the current batch for validating the model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes the model output for current batch, and metrics for current model output.
        predictions = self.model([input_batch], True, None)
        loss = self.compute_loss(target_batch, predictions)
        accuracy = self.compute_accuracy(target_batch, predictions)

        # Computes batch metrics and appends it to main metrics.
        self.validation_loss(loss)
        self.validation_accuracy(accuracy)

    def reset_trackers(self) -> None:
        """Resets states for training and validation trackers before the start of each epoch.

        Resets states for training and validation trackers before the start of each epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss.reset_states()
        self.validation_loss.reset_states()
        self.train_accuracy.reset_states()
        self.validation_accuracy.reset_states()


def main():
    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Version by which the trained model files should be saved as.",
    )
    args = parser.parse_args()

    # Creates an logger object for storing terminal output.
    create_log("train_v{}".format(args.model_version), "logs/cifar_classification")
    add_to_log("")

    # Sets memory limit of GPU if found in the system.
    set_physical_devices_memory_limit()

    # Creates an object for the Train class.
    trainer = Train(args.model_version)

    # Loads model configuration for current model version.
    trainer.load_model_configuration()

    # Loads dataset based on dataset version in the model configuration.
    trainer.load_dataset()

    # Loads model & other utilies for training it.
    trainer.load_model()

    # Generates summary and plot for loaded model.
    trainer.generate_model_summary_and_plot()


if __name__ == "__main__":
    main()
