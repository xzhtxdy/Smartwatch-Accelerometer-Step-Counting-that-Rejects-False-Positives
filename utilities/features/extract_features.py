import os
import numpy as np
import pandas as pd
from utilities.features.calculate_features import get_features
from ..preprocess import read_data_for_one_subject, make_windows, data_filter
from ..parse_config import parameters

data_dir = parameters['data_dir']
subjectNames = parameters['subject_names_train']
activityNames = parameters['activityNames']



def extract_feature_for_subject(subject, win_len, stride, **kwargs):

    data = read_data_for_one_subject(subject, **kwargs)

    first_flag = True
    Features = np.empty(shape=(0, 26))
    for k, activity in enumerate(activityNames):
        array1 = data[activity]
        feature1 = extract_feature_for_exercise_train(array1, win_len, stride)
        feature1 = np.hstack((feature1, np.ones((feature1.shape[0], 1)) * k))
        # if k<=8:
        if k <= 7:
            features1 = np.hstack((feature1, np.ones((feature1.shape[0], 1)) * 0))
        else:
            features1 = np.hstack((feature1, np.ones((feature1.shape[0], 1)) * 1))
        if first_flag:
            Features = features1
            first_flag = False
        else:
            Features = np.vstack((Features, features1))
    return Features

def extract_feature_for_exercise_train(array, win_len, stride):

    first_flag = True
    data_wins = make_windows(array, win_len, stride, dim=array.shape[1])
    for data_win in data_wins:
        if data_win.shape[0] == win_len:
            feature = get_features(data_win)
            if first_flag:
                Features = feature
                first_flag = False
            else:
                Features = np.vstack((Features, feature))
    return Features

def extract_feature_for_exercise(array):

    first_flag = True

    feature = get_features(array)
    if first_flag:
        Features = feature
        first_flag = False
    else:
        Features = np.vstack((Features, feature))
    return Features


def extract_features(window_size, stride, **kwargs):

    for i, subject in enumerate(subjectNames):
        print('Extracting features for subject: ', subject)
        pd_Features = extract_feature_for_subject(subject, window_size, stride, **kwargs)
        pd_Features = pd.DataFrame(pd_Features)
        feature_path = os.path.join(data_dir,'raw', 'data' ,subject, 'Gsenser Data', f'features_{window_size}_{stride}.csv')
        pd_Features.to_csv(feature_path, header=False, index=False, encoding='utf-8')
    return pd_Features


def extract_single_features(array1):

    feature1 = extract_feature_for_exercise(array1)
    feature1 = feature1.reshape(1, -1)

    return feature1
