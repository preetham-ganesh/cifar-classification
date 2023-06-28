# Copyrights (c) Preetham Ganesh.


import os
import sys
import warnings


BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")


from typing import Dict, Any
import tensorflow_datasets as tfds
import cv2

from src.utils import check_directory_path_existence
from src.utils import add_to_log


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

    def download_extract_data(self, split: str) -> None:
        """Downloads the CIFAR-100 dataset, and saves the image and label information.

        Downloads the CIFAR-100 dataset, and saves the image and label information.

        Args:
            split: A string for the name of the current data split.

        Returns:
            None.

        Exceptions:
            OSError: When the directory requested to be deleted is not available.
        """
        # Asserts type & value of the arguments.
        assert isinstance(split, str), "Variable split should be of type 'str'."
        assert split in [
            "train",
            "test",
        ], "Variable split should have 'train' or 'test' as value."

        self.home_directory_path = os.getcwd()

        # Creates the following directory path if it does not exist.
        raw_data_directory_path = check_directory_path_existence(
            "data/raw_data/cifar_100/{}".format(split)
        )

        # Downloads cifar-100 dataset into the corresponding directory for the current data split.
        dataset, info = tfds.load(
            "cifar100",
            split=split,
            with_info=True,
            shuffle_files=True,
            data_dir=raw_data_directory_path,
            as_supervised=True,
        )
        add_to_log("")
        add_to_log("Downloaded the CIFAR-100 dataset for {} split.".format(split))
        add_to_log("")
