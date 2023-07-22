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


from typing import Dict, Any

from src.utils import load_json_file
from src.utils import create_log
from src.utils import add_to_log
from src.utils import set_physical_devices_memory_limit
from src.cifar_classification.model import CifarClassificationCNN
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


def load_model_configuration(model_version: str) -> Dict[str, Any]:
    """Loads the model configuration file for current version.

    Loads the model configuration file for current version.

    Args:
        model_version: A string for the version of the model.

    Returns:
        A dictionary for the loaded model configuration.
    """
    home_directory_path = os.getcwd()
    model_configuration_directory_path = (
        "{}/configs/models/cifar_classification".format(home_directory_path)
    )
    model_configuration = load_json_file(
        "v{}".format(model_version), model_configuration_directory_path
    )
    return model_configuration


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

    # Loads the model configuration file for current version.
    model_configuration = load_model_configuration(args.model_version)
    add_to_log("Finished loading model configuration for current version.")
    add_to_log("")

    # Creates an object for DigitRecognitionCNN class.
    cifar_classification = CifarClassificationCNN(model_configuration)

    # Creates an object for the Dataset class.
    dataset = Dataset(model_configuration)

    # Downloads the CIFAR-100 dataset, and saves the image and label information.
    dataset.download_extract_data("train")

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
