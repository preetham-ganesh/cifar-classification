# Copyrights (c) Preetham Ganesh.

import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Dict, Any

from src.utils import load_json_file


def load_model_configuration(model_version: str) -> Dict[str, Any]:
    """Loads the model configuration file for current version.

    Loads the model configuration file for current version.

    Args:
        model_version: A string for the version of the model.

    Returns:
        A dictionary for the loaded model configuration.
    """
    home_directory_path = os.getcwd()
    model_configuration_directory_path = (
        "{}/configs/models/cifar_classification".format(home_directory_path)
    )
    model_configuration = load_json_file(
        "v{}".format(model_version), model_configuration_directory_path
    )
    return model_configuration
