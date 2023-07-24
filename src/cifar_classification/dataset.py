# Copyrights (c) Preetham Ganesh.


import os

from typing import Dict, Any
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
from typing import List

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

        # Computes no. of examples in each data split.
        self.n_train_examples = len(self.train_data)
        self.n_validation_examples = len(self.validation_data)
        self.n_test_examples = len(self.test_data)

        add_to_log(
            "No. of images in the training data: {}".format(self.n_train_examples)
        )
        add_to_log(
            "No. of images in the validation data: {}".format(
                self.n_validation_examples
            )
        )
        add_to_log(
            "No. of images in the validation data: {}".format(self.n_test_examples)
        )
        add_to_log("")

    def map_label_ids_names(self) -> Dict[str, str]:
        """Creates a dictionary to store the unique label ids & names.

        Creates a dictionary to store the unique label ids & names.

        Args:
            None.

        Returns:
            A dictionary for the unique label ids & names.
        """
        # Iterates across rows in the training data.
        labels = {}
        for index in range(self.n_train_examples):
            # Extracts label id & name at current index.
            id = self.train_data.iloc[index]["label_id"]
            name = self.train_data.iloc[index]["label_name"]

            # Adds the label id & name to the dictionary.
            labels[str(id)] = name
        return labels

    def shuffle_slice_datasets(self) -> None:
        """Converts images id & label id into tensorflow dataset and slices them based on batch size.

        Converts images id & label id into tensorflow dataset and slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Shuffles text in each data split.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.train_data["image_id"]),
                list(self.train_data["label_id"]),
            )
        ).shuffle(self.n_train_examples)
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.validation_data["image_id"]),
                list(self.validation_data["label_id"]),
            )
        ).shuffle(self.n_validation_examples)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                list(self.test_data["image_id"]),
                list(self.test_data["label_id"]),
            )
        ).shuffle(self.n_test_examples)

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.train_dataset = self.train_dataset.batch(
            self.model_configuration["batch_size"], drop_remainder=True
        )
        self.validation_dataset = self.validation_dataset.batch(
            self.model_configuration["batch_size"], drop_remainder=True
        )
        self.test_dataset = self.test_dataset.batch(
            self.model_configuration["batch_size"], drop_remainder=True
        )

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = (
            self.n_train_examples // self.model_configuration["batch_size"]
        )
        self.n_validation_steps_per_epoch = (
            self.n_validation_examples // self.model_configuration["batch_size"]
        )
        self.n_test_steps_per_epoch = (
            self.n_test_examples // self.model_configuration["batch_size"]
        )

        add_to_log(
            "No. of train steps per epoch: {}".format(self.n_train_steps_per_epoch)
        )
        add_to_log(
            "No. of validation steps per epoch: {}".format(
                self.n_validation_steps_per_epoch
            )
        )
        add_to_log(
            "No. of test steps per epoch: {}".format(self.n_test_steps_per_epoch)
        )
        add_to_log("")

    def load_image(self, image_id: str) -> np.ndarray:
        """Loads the PNG image for the current image id as a NumPy array.

        Loads the PNG image for the current image id as a NumPy array.

        Args:
            image_id: A string for the image id at current index.

        Returns:
            A NumPy array for the currently loaded image.
        """
        # Asserts type & value of the arguments.
        assert isinstance(image_id, str), "Variable image_id should be of type 'str'."

        # Loads the PNG image for the current image id as a NumPy array.
        file_path = "{}/data/extracted_data/cifar_100/v{}/images/{}.png".format(
            self.home_directory_path,
            self.model_configuration["dataset_version"],
            image_id,
        )
        image = cv2.imread(file_path)
        return image

    def load_input_target_batches(
        self, image_ids: List[np.ndarray], label_ids: List[np.ndarray]
    ) -> List[tf.Tensor]:
        """Load input & target batch for image & labels ids.

        Load input & target batch for image & labels ids.

        Args:
            image_ids: A list of tensors for the image ids in the current batch.
            label_ids: A list of tensors for the label ids in the current batch.

        Returns:
            A list of tensors for the input & target batches generated from image & label ids.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            image_ids, list
        ), "Variable image_ids should be of type 'list'."
        assert isinstance(
            label_ids, list
        ), "Variable label_ids should be of type 'list'."

        # Iterates across images & labels ids in current batch.
        input_batch, target_batch = list(), list()
        for index in range(self.model_configuration["batch_size"]):
            # Loads the PNG image for the current image id as a NumPy array.
            image = self.load_image(str(image_ids[index]))

            # Appends loaded image & the label id for the image to the current input & target batches.
            input_batch.append(image)
            target_batch.append(int(label_ids[index]))

        # Converts input & target batch lists into tensors.
        input_batch = tf.convert_to_tensor(input_batch, dtype=tf.float32)
        target_batch = tf.convert_to_tensor(target_batch, dtype=tf.int8)

        # Normalizes the input batches from [0, 255] to [0, 1] range
        input_batch = input_batch / 255.0
        return input_batch, target_batch
