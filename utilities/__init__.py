from .features.calculate_features import *
from .feature_selection import *
from .generate_dataset import *
from .parse_config import *
from .evaluate_classifier import *
"""
This package provides the utilities for the project, including functions to:
    1. read the data from the dataset
    2. preprocess the data, including filter the data, normalize the data
    3. make the windows of the data
    4. get the features from the windows
    5. select the features
    6. generate the dataset
    7. save the model
    8. generate the train and test dataset
    9. evaluate the classifier
This package contains the following modules:
    1. parameters.py: the parameters of the dataset
    2. read_data.py: read the data from the dataset
    3. data_filter.py: filter the data
    4. make_windows.py: make the windows of the data
    5. calculate_features.py: get the features from the windows
    6. feature_selection.py: select the features
    7. generate_dataset.py: generate the dataset
    8. save_model.py: save the model
    9. generate_train_test_dataset.py: generate the train and test dataset
    10. evaluate_classifier.py: evaluate the classifier
"""

__all__ = ['parameters',
           'get_features',
           'feature_selection',
           'generate_dataset',
           'generate_train_test_dataset',
           'evaluate_classifier_LOSO']
