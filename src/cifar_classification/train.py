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

from src.utils import load_json_file
from src.utils import create_log
from src.utils import add_to_log
from src.utils import set_physical_devices_memory_limit
from src.cifar_classification.model import Model
from src.cifar_classification.dataset import Dataset


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

    """# Loads model & other utilities for training it.
    cifar_classification.load_model()
    add_to_log("Finished loading model for current model configuration.")
    add_to_log("")

    # Generates summary and plot for loaded model.
    (
        model_summary,
        model_plot_path,
    ) = cifar_classification.generate_model_summary_and_plot()
    add_to_log(model_summary)
    add_to_log("")
    add_to_log("Model plot saved at {}.".format(model_plot_path))
    add_to_log("")"""


if __name__ == "__main__":
    main()
