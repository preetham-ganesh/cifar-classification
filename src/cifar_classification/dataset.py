# Copyrights (c) Preetham Ganesh.


import os

from typing import Dict, Any
import pandas as pd


class Dataset(object):
    """Loads the dataset based on model configuration."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Creates object attributes for the Dataset class.

        Creates object attributes for the Dataset class.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initalizes class variables.
        self.model_configuration = model_configuration

    def load_data(self) -> None:
        """Loads the training, validation, and testing labels created when downloading the dataset.

        Loads the training, validation, and testing labels created when downloading the dataset.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        self.train_data = pd.read_csv(
            "{}/data/extracted_data/cifar_100/v{}/labels/train.csv".format(
                self.home_directory_path, self.model_configuration["dataset_version"]
            )
        )
        self.validation_data = pd.read_csv(
            "{}/data/extracted_data/cifar_100/v{}/labels/validation.csv".format(
                self.home_directory_path, self.model_configuration["dataset_version"]
            )
        )
        self.test_data = pd.read_csv(
            "{}/data/extracted_data/cifar_100/v{}/labels/test.csv".format(
                self.home_directory_path, self.model_configuration["dataset_version"]
            )
        )
