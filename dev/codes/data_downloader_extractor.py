# authors_name = 'Preetham Ganesh'
# project_title = 'Comparison on approaches towards classification of CIFAR-100 dataset'
# email = 'preetham.ganesh2015@gmail.com'


import os
import sys
import logging

import tensorflow_datasets as tfds
import zipfile
import tarfile
import pandas as pd

from utils import create_log
from utils import log_information
from utils import check_directory_path_existence


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def data_download_extractor(data_split: str):
    """
    """
    home_directory_path = os.path.dirname(os.getcwd())

    # Creates the following directory path if it does not exist.
    dataset_directory_path = check_directory_path_existence('data/downloaded_data/cifar-100/{}'.format(data_split))

    # Downloads cifar-100 dataset into the corresponding directory for the current data split.
    _, info = tfds.load(
        'cifar100', split=data_split, with_info=True, shuffle_files=True, data_dir=dataset_directory_path
    )
    log_information('')
    log_information('Downloaded the CIFAR-100 dataset for {} data split.'.format(data_split))
    log_information('')

    #
    n_original_examples = info.splits[data_split].num_examples
    print(info)


def main():
    log_information('')
    data_download_extractor('train')


if __name__ == '__main__':
    major_version = 1
    minor_version = 0
    revision = 0
    global version
    version = '{}.{}.{}'.format(major_version, minor_version, revision)
    create_log('logs/data_download/', 'model_training_testing_v{}.log'.format(version))
    main()
