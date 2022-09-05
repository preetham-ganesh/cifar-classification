# authors_name = 'Preetham Ganesh'
# project_title = 'Comparison on approaches towards classification of CIFAR-100 dataset'
# email = 'preetham.ganesh2015@gmail.com'


import os
import logging
import warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings("ignore")


import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import pandas as pd

from utils import create_log
from utils import log_information
from utils import check_directory_path_existence


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def data_downloader_extractor(data_split: str) -> None:
    """Downloads the CIFAR-100 dataset, and saves the image and label information.

    Args:
        data_split: A string which contains the name of the current data split.
    
    Returns:
        None.
    """
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 
        'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
        'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 
        'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 
        'willow_tree', 'wolf', 'woman', 'worm'
    ]

    # Creates the following directory path if it does not exist.
    downloaded_dataset_directory_path = check_directory_path_existence('data/downloaded_data/{}'.format(data_split))

    # Downloads cifar-100 dataset into the corresponding directory for the current data split.
    dataset, info = tfds.load(
        'cifar100', split=data_split, with_info=True, shuffle_files=True, data_dir=downloaded_dataset_directory_path,
        as_supervised=True
    )
    log_information('')
    log_information('Downloaded the CIFAR-100 dataset for {} data split.'.format(data_split))

    # Creates the following directory paths if it does not exist.
    extracted_images_directory_path = check_directory_path_existence('data/extracted_data/v{}/images'.format(version))
    extracted_labels_directory_path = check_directory_path_existence('data/extracted_data/v{}/labels'.format(version))

    # Computes the number of images already saved in the directory.
    n_images_saved = len(os.listdir(extracted_images_directory_path))

    # Creates an empty dictionary for storing image
    images_information = {'image_id': list(), 'label': list(), 'class': list()}

    # Iterates across examples in the dataset.
    n_original_examples = info.splits[data_split].num_examples
    for (index, (image, label)) in enumerate(dataset.take(n_original_examples)):

        # Converts the image into a NumPy array, and saves it as PNG file.
        cv2.imwrite('{}/{}.png'.format(extracted_images_directory_path, n_images_saved + index), image.numpy())

        # Saves the current image information to the dictionary.
        images_information['image_id'].append(n_images_saved + index)
        images_information['label'].append(label.numpy())
        images_information['class'].append(classes[label.numpy()])
    
    # Converts the dictionary into a dataframe.
    images_information_df = pd.DataFrame(images_information)

    # If the data split name is train, then the whole dataset is saved as a CSV file.
    if data_split == 'train':
        images_information_df.to_csv('{}/{}.csv'.format(extracted_labels_directory_path, data_split), index=False)
    
    # Else, splits the dataframe into 2 equal halves, and saves each of those as CSV files.
    else:
        validation_image_information_df = images_information_df.iloc[: len(images_information_df) // 2]
        test_image_information_df = images_information_df.iloc[len(images_information_df) // 2:]
        validation_image_information_df.to_csv(
            '{}/{}.csv'.format(extracted_labels_directory_path, 'validation'), index=False
        )
        test_image_information_df.to_csv('{}/{}.csv'.format(extracted_labels_directory_path, 'test'), index=False)
    
    log_information('')
    log_information('Extracted the CIFAR-100 dataset for {} data split.'.format(data_split))
    log_information('')
    

def main():
    log_information('')
    data_downloader_extractor('train')
    data_downloader_extractor('test')


if __name__ == '__main__':
    major_version = 1
    minor_version = 1
    revision = 0
    global version
    version = '{}.{}.{}'.format(major_version, minor_version, revision)
    create_log('logs/data_download/', 'model_training_testing_v{}.log'.format(version))
    main()
