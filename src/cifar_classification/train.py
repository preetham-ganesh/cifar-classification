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
            "train_precision": list(),
            "validation_precision": list(),
            "train_recall": list(),
            "validation_recall": list(),
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
        self.train_precision = tf.keras.metrics.Mean(name="train_precision")
        self.validation_precision = tf.keras.metrics.Mean(name="validation_precision")
        self.train_recall = tf.keras.metrics.Mean(name="train_recall")
        self.validation_recall = tf.keras.metrics.Mean(name="validation_recall")

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
        self.model_history["train_precision"].append(
            str(round(self.train_precision.result().numpy(), 3))
        )
        self.model_history["validation_precision"].append(
            str(round(self.validation_precision.result().numpy(), 3))
        )
        self.model_history["train_recall"].append(
            str(round(self.train_recall.result().numpy(), 3))
        )
        self.model_history["validation_recall"].append(
            str(round(self.validation_recall.result().numpy(), 3))
        )

        # Saves the model history dictionary as a JSON file.
        save_json_file(self.model_history, "history", self.reports_directory_path)


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
