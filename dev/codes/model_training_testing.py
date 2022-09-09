# authors_name = 'Preetham Ganesh'
# project_title = 'Comparison on approaches towards classification of CIFAR-100 dataset'
# email = 'preetham.ganesh2021@gmail.com'


import numpy as np

from utils import create_log
from utils import log_information
from utils import load_images_information
from utils import load_dataset_input_target
from utils import save_json_file
from utils import shuffle_slice_dataset
from utils import model_training_validation
from utils import model_testing


def main():
    log_information('')

    # Loads images information for all data splits in the dataset.
    extracted_dataset_version = '1.1.0'
    train_images_information, validation_images_information, test_images_information = load_images_information(
        extracted_dataset_version
    )
    log_information('No. of images in the train data split: {}'.format(len(train_images_information)))
    log_information('No. of images in the validation data split: {}'.format(len(validation_images_information)))
    log_information('No. of images in the test data split: {}'.format(len(test_images_information)))
    log_information('')

    # Converts data split images information into input and target data.
    final_image_size = 32
    n_channels = 3
    classes = list(np.unique(train_images_information['class']))
    train_input_data, train_target_data = load_dataset_input_target(
        list(train_images_information['image_id']), list(train_images_information['label']), final_image_size,
        n_channels, len(classes), extracted_dataset_version
    )
    del train_images_information
    validation_input_data, validation_target_data = load_dataset_input_target(
        list(validation_images_information['image_id']), list(validation_images_information['label']), final_image_size,
        n_channels, len(classes), extracted_dataset_version
    )
    del validation_images_information
    test_input_data, test_target_data = load_dataset_input_target(
        list(test_images_information['image_id']), list(test_images_information['label']), final_image_size, n_channels, 
        len(classes), extracted_dataset_version
    )
    del test_images_information

    # Creates model configuration for training the model.
    batch_size = 32
    model_configuration = {
        'epochs': 100, 'batch_size': batch_size, 'final_image_size': final_image_size, 'n_channels': n_channels,
        'model_version': version, 'learning_rate': 0.001, 'extracted_dataset_version': extracted_dataset_version,
        'train_steps_per_epoch': len(train_input_data) // batch_size,
        'validation_steps_per_epoch': len(validation_input_data) // batch_size,
        'test_steps_per_epoch': len(test_input_data) // batch_size, 'classes': classes,
        'layers_arrangement': [
            'conv2d_0', 'conv2d_1', 'maxpool2d_0', 'dropout_0', 'conv2d_2', 'conv2d_3', 'maxpool2d_1', 'dropout_1',
            'conv2d_4', 'conv2d_5', 'maxpool2d_2', 'dropout_2', 'flatten', 'dense_0', 'dropout_3', 'dense_1', 
            'dropout_4', 'dense_2'
         ],
         'layers_start_index': 0,
         'layers_configuration': {
             'conv2d_0': {
                 'filters': 128, 'kernel': 3, 'padding': 'same', 'activation': 'relu', 'strides': (1, 1),
                 'kernel_initializer': 'glorot_uniform'
             },
             'conv2d_1': {
                 'filters': 128, 'kernel': 3, 'padding': 'same', 'activation': 'relu', 'strides': (1, 1),
                 'kernel_initializer': 'glorot_uniform'
             },
             'maxpool2d_0': {'pool_size': (2, 2), 'strides': (2, 2)},
             'dropout_0': {'rate': 0.4},
             'conv2d_2': {
                 'filters': 256, 'kernel': 3, 'padding': 'same', 'activation': 'relu', 'strides': (1, 1),
                 'kernel_initializer': 'glorot_uniform'
             },
             'conv2d_3': {
                 'filters': 256, 'kernel': 3, 'padding': 'same', 'activation': 'relu', 'strides': (1, 1),
                 'kernel_initializer': 'glorot_uniform'
             },
             'maxpool2d_1': {'pool_size': (2, 2), 'strides': (2, 2)},
             'dropout_1': {'rate': 0.4},
             'conv2d_4': {
                 'filters': 512, 'kernel': 3, 'padding': 'same', 'activation': 'relu', 'strides': (1, 1),
                 'kernel_initializer': 'glorot_uniform'
             },
             'conv2d_5': {
                 'filters': 512, 'kernel': 3, 'padding': 'same', 'activation': 'relu', 'strides': (1, 1),
                 'kernel_initializer': 'glorot_uniform'
             },
             'maxpool2d_2': {'pool_size': (2, 2), 'strides': (2, 2)},
             'dropout_2': {'rate': 0.4},
             'flatten': {},
             'dense_0': {'units': 1024, 'activation': 'relu'},
             'dropout_3': {'rate': 0.4},
             'dense_1': {'units': 1024, 'activation': 'relu'},
             'dropout_4': {'rate': 0.4},
             'dense_2': {'units': len(classes), 'activation': 'softmax'}
        }
    }

    # Saves the model configuration as a JSON file.
    save_json_file(model_configuration, 'model_configuration', 'results/v{}/utils'.format(version))
    log_information('')

    # Shuffles input and target data. Converts into tensorflow datasets.
    validation_dataset = shuffle_slice_dataset(validation_input_data, validation_target_data, batch_size)
    del validation_input_data, validation_target_data
    test_dataset = shuffle_slice_dataset(test_input_data, test_target_data, batch_size)
    del test_input_data, test_target_data
    train_dataset = shuffle_slice_dataset(train_input_data, train_target_data, batch_size)
    del train_input_data, train_target_data
    log_information('Shuffled & Sliced the datasets.')
    log_information('')
    log_information('No. of Training steps per epoch: {}'.format(model_configuration['train_steps_per_epoch']))
    log_information('No. of Validation steps per epoch: {}'.format(model_configuration['validation_steps_per_epoch']))
    log_information('No. of Testing steps: {}'.format(model_configuration['test_steps_per_epoch']))
    log_information('')

    # Trains and validation the model.
    model_training_validation(train_dataset, validation_dataset, model_configuration)
    log_information('')

    # Tests the trained model on test dataset.
    model_testing(test_dataset, model_configuration)
    log_information('')


if __name__ == '__main__':
    major_version = 1
    minor_version = 5
    revision = 1
    global version
    version = '{}.{}.{}'.format(major_version, minor_version, revision)
    create_log('logs', 'model_training_testing_v{}.log'.format(version))
    main()
