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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import time

from model import CifarImageRecognition


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


def load_images_information(extracted_dataset_version: str) -> tuple:
    """Loads extracted images information for all data splits in the dataset.

    Args:
        extracted_dataset_version: A string which contains the version of the extracted data.
    
    Returns:
        A tuple which contains the extracted images information for train, validation and test data splits in the 
            dataset.
    """
    home_directory_path = os.path.dirname(os.getcwd())
    images_information_directory_path = '{}/data/extracted_data/v{}/labels'.format(
        home_directory_path, extracted_dataset_version
    )

    # Reads images information for all data splits in the dataset.
    train_images_information = pd.read_csv('{}/train.csv'.format(images_information_directory_path))
    validation_images_information = pd.read_csv('{}/validation.csv'.format(images_information_directory_path))
    test_images_information = pd.read_csv('{}/test.csv'.format(images_information_directory_path))
    return train_images_information, validation_images_information, test_images_information


def load_preprocess_image(
    image_id: str, final_image_size: int, n_channels: int, extracted_dataset_version: str
) -> tf.Tensor:
    """Loads the image and preprocesses it, for the current image id and other parameters.

    Args:
        image_id: A string which contains name of the current image.
        final_image_size: An integer which contains the size of the final input image.
        n_channels: An integer which contains the number of channels in the read image.
        extracted_dataset_version: A string which contains the version of the extracted data.
    
    Returns:
        A tensor which contains processed image for current image id.
    """
    home_directory_path = os.path.dirname(os.getcwd())

    # Reads image as bytes for current image id. Converts it into RGB format.
    image = tf.io.read_file('{}/data/extracted_data/v{}/images/{}.png'.format(
        home_directory_path, extracted_dataset_version, image_id
    ))
    image = tf.image.decode_jpeg(image, channels=n_channels)

    # Resizes image based on final image size.
    if image.shape[0] > final_image_size or image.shape[1] > final_image_size:
        image = tf.image.resize(image, (final_image_size, final_image_size))
    return image


def load_dataset_input_target(
    data_split_image_ids: list, data_split_labels: list, final_image_size: int, n_channels: int, n_classes: int,
    extracted_dataset_version: str
) -> tuple:
    """Loads current data split's input and target data as tensor.

    Args:
        data_split_image_ids: A list which contains image ids in the current data split.
        data_split_labels: A list which contains labels in the current data split.
        final_image_size: An integer which contains the size of the final input image.
        n_channels: An integer which contains the number of channels in the read image.
        n_classes: An integer which contains the number of classes in the dataset.
        extracted_dataset_version: A string which contains the version of the extracted data.
    
    Returns:
        A tuple which contains tensors for the input and target data.
    """
    input_data, target_data = list(), list()

    # Iterates across image ids and labels in the dataset for current data split.
    for image_id, label in zip(data_split_image_ids, data_split_labels):

        # Loads processed image for current image id.
        input_image = load_preprocess_image(image_id, final_image_size, n_channels, extracted_dataset_version)

        # Creates an one-hot encoded list for current label.
        encoded_label = [1 if index == label else 0 for index in range(n_classes)]

        # Appends image and encoded label for current image to main lists.
        input_data.append(input_image)
        target_data.append(encoded_label)
    
    # Converts list into tensor.
    input_data = tf.convert_to_tensor(input_data, dtype=tf.uint8)
    target_data = tf.convert_to_tensor(target_data, dtype=tf.uint8)
    return input_data, target_data


def shuffle_slice_dataset(input_data: list, target_data: list, batch_size: int) -> tf.data.Dataset:
    """Converts the input data and target data into tensorflow dataset and slices them based on batch size.

    Args:
        input_data: A list which contains the current data split's resized input images.
        target_data: A list which contains the current data split's resized target images.
        batch_size: An integer which contains batch size for slicing the dataset into small chunks.

    Returns:
        A TensorFlow dataset which contains sliced input and target tensors for the current data split.
    """
    # Zip input and output tensors into a single dataset and shuffles it.
    dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data)).shuffle(len(input_data))

    # Slices the combined dataset based on batch size, and drops remainder values.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def process_input_batch(input_batch: tf.Tensor) -> tf.Tensor:
    """Processes input batch normalizing the pixel value range, and type casting them to float32 type.

    Args:
        input_batch: A tensor which contains the input images from the current batch for training the model.

    Returns:
        A tensor which contains the processed input batch.
    """
    # Casts input and target batches to float32 type.
    input_batch = tf.cast(input_batch, dtype=tf.float32)

    # Normalizes the input and target batches from [0, 255] range to [0, 1] range.
    input_batch = input_batch / 255.0
    return input_batch


def loss_function(target_batch: tf.Tensor, predicted_batch: tf.Tensor) -> tf.Tensor:
    """Computes the loss value for the current batch of the predicted values based on comparison with actual values.

    Args:
        target_batch: A tensor which contains the actual values for the current batch.
        predicted_batch: A tensor which contains the predicted values for the current batch.

    Returns:
        A tensor which contains loss for the current batch.
    """
    # Creates the loss object for the Categorical Crossentropy & computes loss using the target and predicted batches.
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    current_loss = loss_object(target_batch, predicted_batch)
    return current_loss


def accuracy_function(target_batch: tf.Tensor, predicted_batch: tf.Tensor):
    """Computes the accuracy value for the current batch of the predicted values based on comparison with actual values.

    Args:
        target_batch: A tensor which contains the actual values for the current batch.
        predicted_batch: A tensor which contains the predicted values for the current batch.

    Returns:
        A tensor which contains accuracy for the current batch.
    """
    # Computes loss for the current batch using actual values and predicted values.
    accuracy = tf.keras.metrics.categorical_accuracy(target_batch, predicted_batch)
    return tf.reduce_mean(accuracy)


@tf.function
def train_step(input_batch: tf.Tensor, target_batch: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer) -> None:
    """Trains the convolutional neural network model using the current input and target training batches. Predicts the 
        output for the current input batch, computes loss on comparison with the target batch, and optimizes the model 
        based on the computed loss.

    Args:
        input_batch: A tensor which contains the input images from the current batch for training the model.
        target_batch: A tensor which contains the target images from the current batch for training and validating the 
            model.
        optimizer: An tensorflow optimizer object which contains optimizing algorithm used improve the performance of 
            the model.

    Returns:
        None.
    """
    # Processes input batch for training the model.
    input_batch = process_input_batch(input_batch)

    # Computes masked images for all input images in the batch, and computes batch loss.
    with tf.GradientTape() as tape:
        predicted_batch = model(input_batch, True)
        batch_loss = loss_function(target_batch, predicted_batch)

    # Computes gradients using loss & model variables. Apply the computed gradients on model variables using optimizer.
    gradients = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Computes accuracy score for the current batch.
    batch_accuracy = accuracy_function(target_batch, predicted_batch)

    # Computes mean for loss and accuracy.
    train_loss(batch_loss)
    train_accuracy(batch_accuracy)


def validation_step(input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
    """Validates the model using the current input and target validation batches.

    Args:
        input_batch: A tensor which contains the input images from the current batch for validating the model.
        target_batch: A tensor which contains the target images from the current batch for validating the model.

    Returns:
        None.
    """
    # Processes input batch for validating the model.
    input_batch = process_input_batch(input_batch)

    # Computes masked images for all input images in the batch.
    predicted_batch = model(input_batch, False)

    # Computes loss & accuracy for the target batch and predicted batch.
    batch_loss = loss_function(target_batch, predicted_batch)
    batch_accuracy = accuracy_function(target_batch, predicted_batch)

    # Computes mean for loss & accuracy.
    validation_loss(batch_loss)
    validation_accuracy(batch_accuracy)


def generate_model_history_plot(split_history_dataframe: pd.DataFrame, metric_name: str, version: str) -> None:
    """Generates plot for model training and validation history.

    Args:
        split_history_dataframe: A Pandas dataframe which contains model training and validation performance history.
        metric_name: A string which contains the current metric name.
        version: A string which contains the current version of the model.

    Returns:
        None.
    """
    # Specifications used to generate the plot, i.e., font size and size of the plot.
    font = {'size': 28}
    plt.rc('font', **font)
    figure(num=None, figsize=(30, 15))

    # Converts train and validation metrics from string format to floating point format.
    epochs = [i for i in range(1, len(split_history_dataframe) + 1)]
    train_metrics = list(split_history_dataframe['train_{}'.format(metric_name)])
    train_metrics = [float(train_metrics[i]) for i in range(len(train_metrics))]
    validation_metrics = list(split_history_dataframe['validation_{}'.format(metric_name)])
    validation_metrics = [float(validation_metrics[i]) for i in range(len(validation_metrics))]

    # Generates plot for training and validation metrics
    plt.plot(epochs, train_metrics, color='orange', linewidth=3, label='train_{}'.format(metric_name))
    plt.plot(epochs, validation_metrics, color='blue', linewidth=3, label='validation_{}'.format(metric_name))

    # Generates the plot for the epochs vs metrics.
    plt.xlabel('epochs')
    plt.xticks(epochs)
    plt.ylabel(metric_name)
    plt.legend(loc='upper left')
    plt.grid(color='black', linestyle='-.', linewidth=2, alpha=0.3)

    # Saves plot using the following path.
    home_directory_path = os.path.dirname(os.getcwd())
    plt.savefig('{}/results/v{}/utils/model_history_{}.png'.format(home_directory_path, version, metric_name))
    plt.close()


def model_training_validation(
    train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, model_configuration: dict
) -> None:
    """Trains and validates the current configuration of the model using the train and validation dataset.

    Args:
        train_dataset: A TensorFlow dataset which contains sliced input and target tensors for the Training data split.
        validation_dataset: A TensorFlow dataset which contains sliced input and target tensors for the Validation data 
            split.
        model_configuration: A dictionary which contains current model configuration details.

    Returns:
        None.
    """
    global model, train_loss, validation_loss, train_accuracy, validation_accuracy
    home_directory_path = os.path.dirname(os.getcwd())

    # Tensorflow metrics which computes the mean of all the elements.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    validation_accuracy = tf.keras.metrics.Mean(name='validation_accuracy')

    # Creates instances for neural network model.
    model = CifarImageRecognition(model_configuration)

    # Creates checkpoint and manager for the neural network model and the optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_configuration['learning_rate'])
    checkpoint_directory_path = '{}/results/v{}/checkpoints'.format(
        home_directory_path, model_configuration['model_version']
    )
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_count = 0
    best_validation_loss = None
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory_path, max_to_keep=3)

    # Compiles the model to print model summary.
    input_dim = (
        model_configuration['final_image_size'], model_configuration['final_image_size'], 
        model_configuration['n_channels']
    )
    _ = model.build((model_configuration['batch_size'], *input_dim))
    log_information(model.summary())
    log_information('')

    # Plots the model and saves it as a PNG file.
    tf.keras.utils.plot_model(
        model.build_graph(), '{}/results/v{}/utils/model_plot.png'.format(
            home_directory_path, model_configuration['model_version']
        ), show_shapes=True, show_layer_names=True, expand_nested=False
    )

    # Creates empty dataframe for saving the model training and validation metrics for the current model.
    split_history_dataframe = pd.DataFrame(
        columns=[
            'epochs', 'train_loss', 'validation_loss', 'train_accuracy', 'validation_accuracy'
        ]
    )
    split_history_dataframe_path = '{}/results/v{}/utils/split_history.csv'.format(
        home_directory_path, model_configuration['model_version']
    )

    # Iterates across epochs for training the neural network model.
    checkpoint_count = 0
    best_validation_loss = None
    for epoch in range(model_configuration['epochs']):
        epoch_start_time = time.time()

        # Resets states for training and validation metrics before the start of each epoch.
        train_loss.reset_states()
        validation_loss.reset_states()
        train_accuracy.reset_states()
        validation_accuracy.reset_states()

        # Iterates across the batches in the train dataset.
        for (batch, (input_batch, target_batch)) in enumerate(
            train_dataset.take(model_configuration['train_steps_per_epoch'])
        ):
            batch_start_time = time.time()

            # Trains the model using the current input and target batch.
            train_step(input_batch, target_batch, optimizer)
            batch_end_time = time.time()
            if batch % 10 == 0:
                log_information('Epoch={}, Batch={}, Train loss={}, Train accuracy={}, Time taken={} sec.'.format(
                    epoch + 1, batch, str(round(train_loss.result().numpy(), 3)), 
                    str(round(train_accuracy.result().numpy(), 3)), round(batch_end_time - batch_start_time, 3)
                ))
        log_information('')

        # Iterates across the batches in the validation dataset.
        for (batch, (input_batch, target_batch)) in enumerate(
            validation_dataset.take(model_configuration['validation_steps_per_epoch'])
        ):
            batch_start_time = time.time()

            # Validates the model using the current input and target batch.
            validation_step(input_batch, target_batch)
            batch_end_time = time.time()
            if batch % 10 == 0:
                log_information(
                    'Epoch={}, Batch={}, Validation loss={}, Validation accuracy={}, Time taken={} sec.'.format(
                        epoch + 1, batch, str(round(validation_loss.result().numpy(), 3)), 
                        str(round(validation_accuracy.result().numpy(), 3)), 
                        round(batch_end_time - batch_start_time, 3)
                    ))
        log_information('')

        # Updates the complete metrics dataframe with the metrics for the current training and validation metrics.
        history_dictionary = {
            'epochs': int(epoch + 1), 'train_loss': str(round(train_loss.result().numpy(), 3)), 
            'validation_loss': str(round(validation_loss.result().numpy(), 3)), 
            'train_accuracy': str(round(train_accuracy.result().numpy(), 3)), 
            'validation_accuracy': str(round(validation_accuracy.result().numpy(), 3))
        }
        split_history_dataframe = split_history_dataframe.append(history_dictionary, ignore_index=True)
        split_history_dataframe.to_csv(split_history_dataframe_path, index=False)
        epoch_end_time = time.time()
        log_information(
            'Epoch={}, Training loss={}, Validation loss={}, Training accuracy={}, Validation accuracy={}, Time '\
                'taken={} sec.'.format(
                    epoch + 1, str(round(train_loss.result().numpy(), 3)), 
                    str(round(validation_loss.result().numpy(), 3)), str(round(train_accuracy.result().numpy(), 3)), 
                    str(round(validation_accuracy.result().numpy(), 3)), round(epoch_end_time - epoch_start_time, 3)
                )
        )

        # If epoch = 1, then best validation loss is replaced with current validation loss, and the checkpoint is saved.
        if best_validation_loss is None:
            checkpoint_count = 0
            best_validation_loss = str(round(validation_loss.result().numpy(), 3))
            manager.save()
            log_information('Checkpoint saved at {}.'.format(checkpoint_directory_path))
        
        # If the best validation loss is higher than current validation loss, the best validation loss is replaced with 
        # current validation loss, and the checkpoint is saved.
        elif best_validation_loss > str(round(validation_loss.result().numpy(), 3)):
            checkpoint_count = 0
            log_information('Best validation loss changed from {} to {}'.format(
                str(best_validation_loss), str(round(validation_loss.result().numpy(), 3))
            ))
            best_validation_loss = str(round(validation_loss.result().numpy(), 3))
            manager.save()
            log_information('Checkpoint saved at {}'.format(checkpoint_directory_path))
        
        # If the best validation loss is not higher than the current validation loss, then the number of times the 
        # model has not improved is incremented by 1.
        elif checkpoint_count <= 4:
            checkpoint_count += 1
            log_information('Best validation loss did not improve.')
            log_information('Checkpoint not saved.')
        
        # If the number of times the model did not improve is greater than 4, then model is stopped from training 
        # further.
        else:
            log_information('Model did not improve after 4th time. Model stopped from training further.')
            log_information('')
            break
        log_information('')
    
    # Generates plots for all metrics in the metrics in the dataframe.
    generate_model_history_plot(split_history_dataframe, 'loss', model_configuration['model_version'])
    generate_model_history_plot(split_history_dataframe, 'accuracy', model_configuration['model_version'])


def model_testing(test_dataset: tf.data.Dataset, model_configuration: dict) -> None:
    """Tests the currently trained model using the test dataset.

    Args:
        test_dataset: A TensorFlow dataset which contains sliced input and target tensors for the Test data split.
        model_configuration: A dictionary which contains current model configuration details.

    Returns:
        None.
    """
    global model, validation_loss, validation_accuracy
    home_directory_path = os.path.dirname(os.getcwd())

    # Tensorflow metrics which computes the mean of all the elements.
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    validation_accuracy = tf.keras.metrics.Mean(name='validation_accuracy')
    validation_loss.reset_states()
    validation_accuracy.reset_states()

    # Creates instances for neural network model.
    model = CifarImageRecognition(model_configuration)

    checkpoint_directory_path = '{}/results/v{}/checkpoints'.format(
        home_directory_path, model_configuration['model_version']
    )
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory_path))

    # Iterates across the batches in the test dataset.
    for (batch, (input_batch, target_batch)) in enumerate(
        test_dataset.take(model_configuration['validation_steps_per_epoch'])
    ):

        # Validates the model using the current input and target batch.
        validation_step(input_batch, target_batch)
    
    log_information('Test loss={}.'.format(str(round(validation_loss.result().numpy(), 3))))
    log_information('Test accuracy={}.'.format(str(round(validation_accuracy.result().numpy(), 3))))
    log_information('')
