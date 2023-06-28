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


import tensorflow_datasets as tfds
import cv2
import pandas as pd

from src.utils import create_log
from src.utils import check_directory_path_existence
from src.utils import add_to_log


def download_extract_dataset(split: str, dataset_version: str) -> None:
    """Downloads the CIFAR-100 dataset, and saves the image and label information.

    Downloads the CIFAR-100 dataset, and saves the image and label information.

    Args:
        split: A string for the name of the current data split.
        dataset_version: A string for version by which the dataset should be saved as.

    Returns:
        None.
    """
    # Asserts type & value of the arguments.
    assert isinstance(split, str), "Variable split should be of type 'str'."
    assert split in [
        "train",
        "test",
    ], "Variable split should have 'train' or 'test' as value."
    assert isinstance(
        dataset_version, str
    ), "Variable dataset_version should be of type 'str'."

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

    # If train is the split, then removes the directories for extracted images & labels.
    home_directory_path = os.getcwd()
    if split == "train":
        # Removes the directories for images & labels that were previously extracted.
        try:
            os.rmdir(
                "{}/data/extracted_data/cifar_100/v{}/images".format(
                    home_directory_path, dataset_version
                )
            )
            os.rmdir(
                "{}/data/extracted_data/cifar_100/v{}/labels".format(
                    home_directory_path, dataset_version
                )
            )
            add_to_log("Images & labels that were previously extracted, will deleted.")
        except OSError:
            add_to_log("Images & labels were not previously extracted.")
        add_to_log("")

    # Creates the following directory paths if it does not exist.
    extracted_images_directory_path = check_directory_path_existence(
        "data/extracted_data/cifar_100/v{}/images".format(dataset_version)
    )
    extracted_labels_directory_path = check_directory_path_existence(
        "data/extracted_data/cifar_100/v{}/labels".format(dataset_version)
    )

    # Computes the number of images already saved in the directory.
    n_images_saved = len(os.listdir(extracted_images_directory_path))

    # Creates an empty dictionary for storing image
    images_information = {"image_id": list(), "labels": list(), "label_id": list()}

    # A list for the labels in the cifar100 dataset.
    labels = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "cra",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]

    # Iterates across examples in the dataset.
    n_examples = info.splits[split].num_examples
    for index, (image, label_id) in enumerate(dataset.take(n_examples)):
        # Converts the image into a NumPy array, and saves it as PNG file.
        cv2.imwrite(
            "{}/{}.png".format(extracted_images_directory_path, n_images_saved + index),
            image.numpy(),
        )

        # Saves the current image information to the dictionary.
        images_information["image_id"].append(n_images_saved + index)
        images_information["label_id"].append(label_id.numpy())
        images_information["labels"].append(labels[label_id.numpy()])
        if index == 10:
            break

    # Converts the dictionary into a dataframe.
    images_information = pd.DataFrame(images_information)

    # If the data split is train, then the whole dataset is saved as a CSV file.
    if split == "train":
        images_information.to_csv(
            "{}/{}.csv".format(extracted_labels_directory_path, split), index=False
        )

    # Else, splits the dataframe into 2 equal halves, and saves each of those as CSV files.
    else:
        validation_image_information = images_information.iloc[
            : len(images_information) // 2
        ]
        test_image_information = images_information.iloc[len(images_information) // 2 :]
        validation_image_information.to_csv(
            "{}/{}.csv".format(extracted_labels_directory_path, "validation"),
            index=False,
        )
        test_image_information.to_csv(
            "{}/{}.csv".format(extracted_labels_directory_path, "test"), index=False
        )
    add_to_log("")
    add_to_log("Extracted the CIFAR-100 dataset for {} data split.".format(split))
    add_to_log("")


def main():
    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dv",
        "--dataset_version",
        type=str,
        required=True,
        help="Version by which the downloaded & extracted dataset should be saved as.",
    )
    args = parser.parse_args()

    # Creates an logger object for storing terminal output.
    create_log(
        "download_dataset_v{}".format(args.dataset_version), "logs/cifar_classification"
    )
    add_to_log("")

    # Downloads the CIFAR-100 dataset, and saves the image and label information.
    download_extract_dataset("train", args.dataset_version)
    # download_extract_dataset("test", args.dataset_version)


if __name__ == "__main__":
    main()
