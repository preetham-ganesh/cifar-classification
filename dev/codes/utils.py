# authors_name = 'Preetham Ganesh'
# project_title = 'Comparison on approaches towards classification of CIFAR-100 dataset'
# email = 'preetham.ganesh2021@gmail.com'


import os
import logging
import warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings("ignore")


import json
import tensorflow as tf
import cv2


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def check_directory_path_existence(directory_path: str) -> str:
    """Creates the absolute path for the directory path given in argument if it does not already exist.

    Args:
        directory_path: A string which contains the directory path that needs to be created if it does not already 
            exist.

    Returns:
        A string which contains the absolute directory path.
    """
    # Creates the following directory path if it does not exist.
    home_directory_path = os.path.dirname(os.getcwd())
    absolute_directory_path = '{}/{}'.format(home_directory_path, directory_path)
    if not os.path.isdir(absolute_directory_path):
        os.makedirs(absolute_directory_path)
    return absolute_directory_path


def create_log(logger_directory_path: str, log_file_name: str) -> None:
    """Creates an object for logging terminal output.

    Args:
        logger_directory_path: A string which contains the location where the log file should be stored.
        log_file_name: A string which contains the name for the log file.

    Returns:
        None.
    """
    # Checks if the following path exists.
    logger_directory_path = check_directory_path_existence(logger_directory_path)

    # Create and configure logger
    logging.basicConfig(
        filename='{}/{}'.format(logger_directory_path, log_file_name), format='%(asctime)s %(message)s', filemode='w'
    )
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def log_information(log: str) -> None:
    """Saves current log information, and prints it in terminal.

    Args:
        log: A string which contains the information that needs to be printed in terminal and saved in log.

    Returns:
        None.

    Exception:
        NameError: When the logger is not defined, this exception is thrown.
    """
    try:
        logger.info(log)
    except NameError:
        _ = ''
    print(log)


def save_json_file(dictionary: dict, file_name: str, directory_path: str) -> None:
    """Converts a dictionary into a JSON file and saves it for future use.

    Args:
        dictionary: A dictionary which needs to be saved.
        file_name: A string which contains the name with which the file has to be saved.
        directory_path: A string which contains the path where the file needs to be saved.

    Returns:
        None.
    """
    # Creates the following directory path if it does not exist.
    directory_path = check_directory_path_existence(directory_path)

    # Saves the dictionary or list as a JSON file at the file path location.
    file_path = '{}/{}.json'.format(directory_path, file_name)
    with open(file_path, 'w') as out_file:
        json.dump(dictionary, out_file, indent=4)
    out_file.close()
    log_information('{} file saved successfully at {}.'.format(file_name, file_path))


def load_json_file(file_name: str, directory_path: str) -> dict:
    """Loads a JSON file into memory based on the file_name.

    Args:
        file_name: A string which contains the name of the of the file to be loaded.
        directory_path: A string which contains the location where the directory path exists.

    Returns:
        A dictionary which contains the JSON file.
    """
    file_path = '{}/{}.json'.format(directory_path, file_name)
    with open(file_path, 'r') as out_file:
        dictionary = json.load(out_file)
    out_file.close()
    return dictionary


def load_preprocess_image(image_id: str, final_image_size: int, n_channels: int) -> tf.Tensor:
    """Loads the image and preprocesses it, for the current image id and other parameters.

    Args:
        image_id: A string which contains name of the current image.
        final_image_size: An integer which contains the size of the final input image.
        n_channels: An integer which contains the number of channels in the read image.
    
    Returns:
        A tensor which contains processed image for current image id.
    """
    home_directory_path = os.path.dirname(os.getcwd())

    # Reads image as bytes for current image id. Converts it into RGB format.
    image = tf.io.read_file('{}/data/extracted_data/images/{}.png'.format(home_directory_path, image_id))
    image = tf.image.decode_jpeg(image, channels=n_channels)

    # Resizes image based on final image size.
    if image.shape[0] > final_image_size or image.shape[1] > final_image_size:
        image = tf.image.resize(image, (final_image_size, final_image_size))
    return image
