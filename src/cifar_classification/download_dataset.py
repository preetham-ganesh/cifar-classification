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

    # Downloads the CIFAR-100 dataset, and saves the image and label information.
    download_extract_dataset("train", args.dataset_version)
    download_extract_dataset("test", args.dataset_version)


if __name__ == "__main__":
    main()
